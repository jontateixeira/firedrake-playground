#!/home/john/opt/firedrake/bin/python3
# -*- coding: utf-8 -*-

"""
Solves the two phases flow through porous media problem in a full implicit
scheme.

Strong form:

    - 2*mu*D(u) + mu*K^(-1)*u + grad(p) = 0         in Omega
                                 div(u) = 0         in Omega
   dc/dt + div(c*u) - div(Diff*grad(c)) = f - k*c^n in Omega

where,

    D(u) = 2*sym(grad(x)).


Weak form:

Find u, p, c in W, such that,

(2*mu*D(v),D(u)) - (p,div(v)) 
    + (mu*K^(-1)*u,v) 
    - ({2*mu*D(u)*n},[v])_S 
    - (Beta*[u],{2*mu*D(v)})_S = (v,p*n)_N             on Omega

                   (q, div(u)) = 0                     on Omega

(r,(c - c0)) + dt*([r],[ũ*c_])_S - dt*(grad(r),u*c_) 
    + dt*(r,ũ.n*cIn)_Inlet + dt*(r,c_*ũ)_s 
    + dt*(Diff*grad(c_),grad(r)) 
    - dt*([r,n],{Diff*grad(c_)})_S 
    + eps*({Diff*fd.grad(phi)},[c_mid, n])_S
    + {gamma/h_E}*([r, n], [c_, n])_S = 0              on Omega

for all v, q, r in W'.

Model problem:

 ----------4----------
 |                   |
 1                   2
 |                   |
 ----------3----------

Initial Conditions:
u(x, 0) = 0 in Omega
p(x, 0) = 0 in Omega
c(x, 0) = 0 in Omega

Boundary Conditions:
p(x, t) = pbar      on Gamma_2
u(x, t) = noflow    on Gamma_{3, 4} if u.n < 0
u(x, t) = qbar      on Gamma_1
c(x, t) = cIn       on Gamma_1 if u.n < 0
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

# SB parameters
k_cavern, k_porous = 1e-8, 1e-13     # permeability [m2]
phi = [1., 0.2]                     # effective porosity [-]
wx, wy = 0.25, 0.35
f = fd.Constant((0., rho*0.))


# reaction/transport parameter
Dm = 1e-5                     # modelecular diffusivity
alphaL = 2e-3                 # longitudinal dispersivity
alphaT = 1e-3                 # transverse dispersivity


# boundary conditions
inlet = LEFT
v_inlet = fd.Constant((0.05, .0))

outlet = RIGHT
p_out = fd.Constant(1e5)            # Pressure [Pa]

noslip = [TOP, BOTTOM]
v_noslip = fd.Constant((.0, .0))

cIn = 1.0

# problem parameters
Beta = - 1              # [+1,-1] non-symmetric/symmetric problem
gama0 = 100             # stabilization parameter (sufficient large)
k = 3
eps = +1                # [+1,0,-1] non-symmetric,Incomplete,Symmetric problem
tol = 1e-15
verbose = False
freq_res = 100
nsteps = 1000        # number of time steps
sim_time = 1200.0     # simulation time


# %%
# Process section (simulation)
# -----------------------------

# 2) Define mesh
print('* define mesh')
mesh = fd.RectangleMesh(nx * n, ny * n, Lx, Ly, quadrilateral=quad_mesh)

if verbose:
    fd.triplot(mesh)
    plt.legend()
    plt.savefig("plots/sb_adr_mesh.png")

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
    plt.savefig("plots/sb_adr_doamins.png")


# 3) Setting problem (FunctionSpace, Init.Bound.Condition, VariationalForms)

# 3.1) Set Function spaces
if quad_mesh:
    RT1 = fd.FunctionSpace(mesh, "RTCF", order)
    DG0 = fd.FunctionSpace(mesh, "DQ", order - 1)
    DG1 = fd.FunctionSpace(mesh, "DQ", order)
else:
    RT1 = fd.FunctionSpace(mesh, "RT", order)
    DG0 = fd.FunctionSpace(mesh, "DG", order - 1)
    DG1 = fd.FunctionSpace(mesh, "DG", order)

# Create mixed function spaces
# W = fd.MixedFunctionSpace([RT1, DG0])
W = RT1 * DG0


# Others function space
P1 = fd.FunctionSpace(mesh, "CG", order)
P1v = fd.VectorFunctionSpace(mesh, "CG", order)


# 3.1) Define trial and test functions
# SB
u, p = fd.TrialFunctions(W)    # H1_gD(Omega), L2(Omega)
v, q = fd.TestFunctions(W)     # H1_0(Omega), L2(Omega)

# transport
phi = fd.TestFunction(DG1)


# 3.2) Set initial conditions
# ---- previous solution
# concentrations
c0 = fd.Function(DG1, name="conc0")
c = fd.Function(DG1, name="conc")


# 3.3) set boundary conditions
# SB
bcF = [fd.DirichletBC(W.sub(0), v_noslip, noslip, method="geometric"),
       fd.DirichletBC(W.sub(0), v_inlet, inlet)]

bcT = fd.DirichletBC(DG1, cIn, inlet, method='geometric')

# ----------------------
# 3.4) Variational Form
# ----------------------
# SB
# coefficients
mu = fd.Constant(mu)
rho = fd.Constant(rho)

n = fd.FacetNormal(mesh)
mul = fd.Function(DG0)
mul.interpolate(fd.conditional(logical, mueff[0], mueff[1]))


def D(x):
    return 2*fd.sym(fd.grad(x))


# variational form (SB)
a = 2*mu*fd.inner(D(v), D(u))*fd.dx - p*fd.div(v)*fd.dx + \
    mu*ikm*fd.inner(u, v)*fd.dx + q*fd.div(u)*fd.dx - \
    2*mu*fd.inner(fd.avg(D(u)*n), fd.jump(v))*fd.dS - \
    2*mu*Beta*fd.inner(fd.avg(D(v)*n), fd.jump(u))*fd.dS

L = fd.inner(f, v)*fd.dx + p_out*fd.inner(v, n)*fd.ds(outlet)


# -------------------------------------------------------------------------
# transport
# coefficients
dt = sim_time / nsteps
Dt = fd.Constant(dt)
c_mid = 0.5 * (c + c0)  # Crank-Nicolson timestepping

# stability
gamma = fd.Constant(2*(mesh.geometric_dimension()+1) /
                    mesh.geometric_dimension())
h_E = fd.sqrt(2) * fd.CellVolume(mesh) / fd.CellDiameter(mesh)

# advective velocity
vel = fd.Function(RT1)
vnorm = fd.sqrt(fd.dot(vel, vel))

# Diffusion tensor
Diff = fd.Identity(mesh.geometric_dimension())*(Dm + alphaT*vnorm) + \
    (alphaL-alphaT)*fd.outer(vel, vel)/vnorm

# upwind term
vn = 0.5*(fd.dot(vel, n) + abs(fd.dot(vel, n)))

# weak form (transport)
# transient term
F_t = phi*(c - c0)*fd.dx

# Diffusion term
F_d = Dt*(fd.inner(Diff*fd.grad(c_mid), fd.grad(phi))*fd.dx
          - fd.inner(fd.jump(phi, n),
                     fd.avg(Diff*fd.grad(c_mid)))*fd.dS
          + eps*fd.inner(fd.avg(Diff*fd.grad(phi)),
                         fd.jump(c_mid, n))*fd.dS
          + fd.avg(gamma/h_E) *
          fd.inner(fd.jump(phi, n), fd.jump(c_mid, n))*fd.dS)

# advection form
F_a = Dt*((phi('+') - phi('-'))*(vn('+')*c_mid('+') -
                                 vn('-')*c_mid('-'))*fd.dS
          - fd.inner(fd.grad(phi), vel*c_mid)*fd.dx
          + fd.conditional(fd.dot(vel, n) < 0, phi *
                           fd.dot(vel, n)*cIn, 0.0)*fd.ds
          + phi*c_mid*vn*fd.ds)
# full weak form
F = F_t + F_a + F_d

# %%
# 4) Solve problem
# -----------------
t = 0.0
it = 0
cfl_conditional = fd.conditional(fd.lt(np.abs(vnorm), tol), 1/tol, h_E/vnorm)
outfile = fd.File("plots/sb_ade.pvd")

x = fd.Function(W)
vD = fd.Function(P1v, name="Velocity Darcy")
c0.assign(0.0)

# initialize timestep
while t < sim_time:
    # move next time step
    t += dt
    print("* iteration= {:4d}, dtime= {:8.6f}, time={:8.6f}".format(it, dt, t))

    # SB
    fd.solve(a == L, x, bcs=bcF)
    vel.assign(x.sub(0))
    CFL = dt/fd.interpolate(cfl_conditional, DG0).dat.data.min()
    print('vel = {}; CFL = {}'.format(
        np.sqrt(np.sum(vel.dat.data*vel.dat.data)).max(), CFL))

    # tracer
    fd.solve(F == 0, c)

    # update sol.
    c0.assign(c)
    it += 1

    # acochambramento c (c<1) and (c>0)
    val = c0.dat.data
    if np.any((val > 1.0) + (val < np.sqrt(tol))):
        val[val > 1.0] = 1.
        val[val < np.sqrt(tol)] = 0.
        c0.dat.data[:] = val

    # %%
    # 5) print results
    if it % freq_res == 0:
        if verbose:
            contours = fd.tripcolor(c0)
            plt.xlabel(r'$x$')
            plt.ylabel(r'$y$')
            plt.colorbar(contours)
            plt.show()

        v, p = x.split()
        p.rename("pressure")
        v.rename("velocidade")
        c0.rename("concentration")
        outfile.write(p, v, c0, time=t)

print('* normal termination.')
