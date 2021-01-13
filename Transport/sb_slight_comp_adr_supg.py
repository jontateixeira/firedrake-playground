#!/home/john/opt/firedrake/bin/python3
# -*- coding: utf-8 -*-
"""
Solves the unsteady state advection-diffusion-reaction problem, using SUPG

Strong form (SF):

    - 2*mu*D(u) + mu*K^(-1)*u + grad(p) = 0         in Omega
                                 div(u) = 0         in Omega
          dc/dt + div(c*u) = div(D*grad(c)) - k*c^n + f

The problem is either advection- or diffusion-dominated, depending on the ratio
u*h/D, where h is the characteristic length scale.

Weak form:

Find u, p, c in W, such that,

(2*mu*D(v),D(u)) - (p,div(v)) 
    + (mu*K^(-1)*u,v) 
    - ({2*mu*D(u)*n},[v])_S 
    - (Beta*[u],{2*mu*D(v)})_S = (v,p*n)_N              on Omega

                    (q, div(u)) = 0                     on Omega

(w, dc/dt) + (w, u.grad(c)) + (grad(w),D*grad(c)) 
            + (w, k*c^n) + SUPG = (w, f)                on Omega

where,
        SUPG  = (grad(w),tau*res*u)
          res = dc/dt + (w, u.grad(c)) + (grad(w),D*grad(c)) 
                   + (w, k*c^n) - (w, f)
          tau = h_mesh/(2*||u||)

for all w in W'.

Model problem:

 -----4-----
 |         |
 1  --> u  2
 |         |
 -----3-----

Initial Conditions:
u(x, 0) = 0 in Omega
p(x, 0) = 0 in Omega
c(x, 0) = 0 in Omega

Boundary Conditions:
p(x, t) = pbar      on Gamma_2
u(x, t) = noflow    on Gamma_{3, 4} if u.n < 0
u(x, t) = qbar      on Gamma_1
c(x, t) = cbar      on Gamma_1
"""
# %%
# 0) importing modules
import matplotlib.pyplot as plt
import numpy as np
import firedrake as fd
print('* loaded modules')

# %%
# 0.1) setting constants
LEFT = 1
RIGHT = 2
BOTTOM = 3
TOP = 4

# %%
# 1) Problem Parameters:
# -----------------------
print('* setting problem parameters')

# domain parameters
order = 1
n = 5
nx = 30
ny = 5
Lx = 60                             # m
Ly = 10                             # m
quad_mesh = True


# materials parameters
# fluid parameters
rho = 1000.                 # fluid density [kg/m3]
mu = 1.0e-3                 # fluid viscosity [N.s/m2 ~ Pa.s]
mueff = [mu, mu]            # effective viscosity
c_t = 0e-10                  # total compressibility [1/Pa]


# SB parameters
k_cavern, k_porous = 1e-8, 1e-13    # permeability [m2]
phi = [1., 0.2]                     # effective porosity [-]
wx, wy = 0.25, 0.35
f = fd.Constant((0., rho*0.))


# reaction/transport parameter
Dm = 1e-5                     # modelecular diffusivity
alphaL = 2e-3                 # longitudinal dispersivity
alphaT = 1e-3                 # transverse dispersivity
K = 0e-3                    # reaction rate

# boundary conditions
inlet = LEFT
v_inlet = fd.Constant((0.05, .0))

outlet = RIGHT
p_out = fd.Constant(1e5)            # Pressure [Pa]

noslip = [TOP, BOTTOM]
v_noslip = fd.Constant((.0, .0))

cIn = 1.0
s = 0.0

# problem parameters
tol = 1e-15
verbose = False
freq_res = 25
CFL = 0.75
dt = 1.2        # time steps size (initial)
sim_time = 400.0     # simulation time


add_shock_term = True
beta = 9e-1  # 2.0


# %%
# Process section (simulation)
# -----------------------------

# 2) Define mesh
print('* define mesh')
mesh = fd.RectangleMesh(nx * n, ny * n, Lx, Ly, quadrilateral=quad_mesh)

if verbose:
    fd.triplot(mesh)
    plt.legend()
    plt.savefig("plots/sb_adr_supg_mesh.png")

# Define subdomains
x = fd.SpatialCoordinate(mesh)
logical = fd.And(fd.And(x[0] > Lx*wx, x[1] > Ly*wy),
                 fd.And(x[0] < Lx*(1-wx), x[1] < Ly*(1-wy)))
square = fd.conditional(logical, 1/k_cavern, 1/k_porous)


ikm = fd.Function(fd.FunctionSpace(mesh, "DG", 0))
ikm.interpolate(square)
if verbose:
    contours = fd.tripcolor(ikm, cmap="viridis")
    cbar = plt.colorbar(contours, aspect=10)
    cbar.set_label("Porous Domain")
    plt.savefig("plots/sb_adr_supg_doamins.png")

# 3) Setting problem (FunctionSpace, Init.Bound.Condition, VariationalForms)

# 3.1) Set Function spaces
V = fd.VectorFunctionSpace(mesh, "CG", order + 1)
P = fd.FunctionSpace(mesh, "CG", order)

# Define function space for system of concentrations (transport)
X = fd.FunctionSpace(mesh, "CG", order)

# Others function space
DG0 = fd.FunctionSpace(mesh, "DG", 0)

# Create mixed function spaces
W = V*P
# W = fd.MixedFunctionSpace([V, P])

# 3.1) Define trial and test functions
# SB
u, p = fd.TrialFunctions(W)
v, q = fd.TestFunctions(W)

# transport
w = fd.TestFunction(X)


# 3.2) Set initial conditions
# ---- previous solution
# SB
w0 = fd.Function(W, name="pv0")
(vD0, p0) = w0.split()

# concentrations
c0 = fd.Function(X, name="conc0")
c = fd.Function(X, name="conc")


# 3.3) set boundary conditions
# SB
bcs = [fd.DirichletBC(W.sub(0), v_noslip, noslip),
       fd.DirichletBC(W.sub(0), v_inlet, inlet),
       fd.DirichletBC(W.sub(1), p_out, outlet)]

# Transport
t_bc = fd.DirichletBC(X, cIn, inlet)


# ----------------------
# 3.4) Variational Form
# ----------------------
# SB
# coefficients
mu = fd.Constant(mu)
rho = fd.Constant(rho)
c_t = fd.interpolate(fd.conditional(logical, phi[0]*c_t, c_t*phi[1]), DG0)
idt = fd.Constant(1 / dt)

n = fd.FacetNormal(mesh)
mul = fd.Function(DG0)
mul.interpolate(fd.conditional(logical, mueff[0], mueff[1]))

# variational form (SB)
a = mul*fd.inner(fd.grad(u), fd.grad(v))*fd.dx - p*fd.div(v)*fd.dx + \
    fd.div(u)*q*fd.dx + idt*c_t*p*q*fd.dx + mu*ikm*fd.inner(u, v)*fd.dx
L = p_out*fd.inner(v, n)*fd.ds(outlet) + \
    fd.inner(f, v)*fd.dx + idt*c_t*p0*q*fd.dx

# -------------------------------------------------------------------------
# transport
# coefficients
K = fd.Constant(K)
Dt = fd.Constant(dt)
c_mid = 0.5 * (c + c0)  # Crank-Nicolson timestepping

# advective velocity
vel = fd.Function(V, name='velocity')
vnorm = fd.sqrt(fd.dot(vel, vel))

# Diffusion tensor
Diff = fd.Identity(mesh.geometric_dimension())*(Dm + alphaT*vnorm) + \
    (alphaL-alphaT)*fd.outer(vel, vel)/vnorm

# source term
fc = fd.Constant(s)


# weak form (transport)
F1 = w*(c - c0)*fd.dx + Dt*(w*fd.dot(vel, fd.grad(c_mid))
                            + fd.dot(fd.grad(w),
                                     Diff*fd.grad(c_mid)) + w*K*c_mid
                            - fc*w)*fd.dx  # - Dt*h_n*w*fd.ds(outlet)

# strong form
R = (c - c0) + Dt*(fd.dot(vel, fd.grad(c_mid)) -
                   fd.div(Diff*fd.grad(c_mid)) + K*c_mid - fc)


# *** Adding SUPG stabilizing and shock cap. terms ***
# SUPG stabilisation parameters
h = fd.sqrt(2)*fd.CellVolume(mesh) / fd.CellDiameter(mesh)
# h = fd.CellSize(mesh)
tau = h / (2.0 * vnorm)
# tau = fd.pow(1/(0.5*Dt) + 2.0*vnorm/h + 4*Diff/fd.pow(h, 2.0), -1)

# Residual and stabilizing terms
F1 += tau*fd.dot(vel, fd.grad(w)) * R * fd.dx

# shock capturing parameters
if add_shock_term:
    cnorm = fd.sqrt(fd.dot(fd.grad(c0), fd.grad(c0)))
    vshock = fd.conditional(fd.gt(cnorm, tol),
                            beta*h / (2*cnorm),  # *cnorm,
                            fd.Constant(0.))

    # shock terms
    # F1 += vshock*fd.dot(fd.grad(w), fd.grad(c_mid))*fd.dx
    F1 += vshock*np.abs(R)*fd.inner(fd.grad(w), fd.grad(c_mid))*fd.dx

    shock = fd.interpolate(vshock*np.abs(R), DG0)
    shock.rename('vshock')

F = F1

prob = fd.NonlinearVariationalProblem(F, c, bcs=t_bc)
transport = fd.NonlinearVariationalSolver(prob)


# %%
# 4) Solve problem
# -----------------
t = 0.0
it = 0
dt0 = dt
cfl_conditional = fd.conditional(fd.lt(np.abs(vnorm), tol), 1/tol, h/vnorm)
outfile = fd.File("plots/sb_t_supg.pvd")

sol = fd.Function(W)
c0.assign(0.)
# initialize timestep
while t < sim_time:
    # move next time step
    print("* iteration= {:4d}, dtime= {:8.6f}, time={:8.6f}".format(it, dt, t))

    # SB
    idt.assign(1 / dt)
    fd.solve(a == L, sol, bcs=bcs)

    vel.assign(sol.sub(0))
    Pe = vnorm * h / (2.0 * fd.det(Diff))
    numCFL = dt/fd.interpolate(cfl_conditional, DG0).dat.data.min()
    print('Peclet = {}; CFL = {}; vel. = {}'.format(fd.interpolate(
        Pe, DG0).dat.data.mean(), numCFL,
        np.sqrt(np.sum(vel.dat.data*vel.dat.data, axis=1)).max()))

    # tracer
    transport.solve()

    # update sol.
    res = fd.norm(fd.interpolate(R, DG0))
    print('Residual = {}'.format(res))

    dt = CFL*fd.interpolate(cfl_conditional, DG0).dat.data.min()
    dt = np.min([sim_time-t, dt]) if (sim_time-t) > dt else dt0
    Dt.assign(dt)

    p0.assign(sol.sub(1))
    c0.assign(c)

    # # acochambramento c (c<1) and (c>0)
    # val = c0.dat.data
    # if np.any((val > 1.0) + (val < np.sqrt(tol))):
    #     val[val > 1.0] = 1.
    #     val[val < np.sqrt(tol)] = 0.
    #     c0.dat.data[:] = val

    # %%
    # 5) print results
    if it % freq_res == 0:
        if verbose:
            contours = fd.tripcolor(c0)
            plt.xlabel(r'$x$')
            plt.ylabel(r'$y$')
            plt.colorbar(contours)
            plt.show()

        v, p = sol.split()
        p.rename("pressure")
        v.rename("velocidade")
        c0.rename("concentration")
        if add_shock_term:
            shock.assign(fd.interpolate(vshock*np.abs(R), DG0))
            outfile.write(p, v, c0, shock, time=t)
        else:
            outfile.write(p, v, c0, time=t)  # tracer

    # move next time step
    t += dt
    it += 1

print('* normal termination.')
