# -*- coding: utf-8 -*-

"""
Solves unsteady state advection-diffusion-reaction problem.

Strong form:

   dc/dt + div(c*u) - div(Diff*grad(c)) = f         in Omega

Weak form:

Find c in W, such that,

(r,(c - c0)) + dt*([r],[ũ*c_])_S - dt*(grad(r),u*c_)
    + dt*(r,ũ.n*cIn)_Inlet + dt*(r,c_*ũ)_s
    + dt*(Diff*grad(c_),grad(r))
    - dt*([r,n],{Diff*grad(c_)})_S
    + eps*({Diff*fd.grad(phi)},[c_mid, n])_S
    + {gamma/h_E}*([r, n], [c_, n])_S + (r,K*c)= 0              on Omega

for all r in W'.

Model problem:

 ----------4----------
 |                   |
 1     ----->        2
 |                   |
 ----------3----------

Initial Conditions:
c(x, 0) = 0 in Omega

Boundary Conditions:
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
n = 2
nx = 30
ny = 5
Lx = 1.0                             # m
Ly = 0.4                             # m
quad_mesh = True


# materials parameters
# reaction/transport parameter
Dm = 1e-9                   # diffusion (m²/s)
d_l = 0.0                   # longitidinal dispersion (m)
d_t = 0.0                   # transversal dispersion (m)
K = 0.00                    # reaction rate (mol/m³/s)


# boundary conditions
inlet = LEFT
v_inlet = fd.Constant((1.0e-6, .0))
cIn = 1.0                   # injection conc

# problem parameters
eps = +1                # [+1,0,-1] non-symmetric,Incomplete,Symmetric problem
tol = 1e-15
verbose = False
freq_res = 5
sim_time = 1.0e6     # simulation time

ptimes = [2e5, 4e5, 6e5, 8e5, 1e6]


# %%
# Process section (simulation)
# -----------------------------

# 2) Define mesh
print('* define mesh')
mesh = fd.RectangleMesh(nx * n, ny * n, Lx, Ly, quadrilateral=quad_mesh)

if verbose:
    fd.triplot(mesh)
    plt.legend()
    plt.savefig("plots/adr_dg_mesh.png")

# 3) Setting problem (FunctionSpace, Init.Bound.Condition, VariationalForms)

# 3.1) Set Function spaces
if quad_mesh:
    DG1 = fd.FunctionSpace(mesh, "DQ", order)
    vDG1 = fd.VectorFunctionSpace(mesh, "DQ", order)
else:
    DG1 = fd.FunctionSpace(mesh, "DG", order)
    vDG1 = fd.FunctionSpace(mesh, "DG", order)


# 3.1) Define trial and test functions
phi = fd.TestFunction(DG1)
c = fd.TrialFunction(DG1)


# 3.2) Set initial conditions
# ---- previous solution
# concentrations
c0 = fd.Function(DG1, name="c")


# 3.3) set boundary conditions
bc = fd.DirichletBC(DG1, cIn, inlet, method='geometric')

# ----------------------
# 3.4) Variational Form
# ----------------------
# coefficients
dt = np.sqrt(tol)
Dt = fd.Constant(dt)
c_mid = 0.5 * (c + c0)  # Crank-Nicolson timestepping
n = fd.FacetNormal(mesh)

# stability
gamma = fd.Constant(2*(mesh.geometric_dimension()+1) /
                    mesh.geometric_dimension())
h_E = fd.sqrt(2) * fd.CellVolume(mesh) / fd.CellDiameter(mesh)

# advective velocity
velocity = fd.Function(vDG1)
velocity.interpolate(v_inlet)
vnorm = fd.sqrt(fd.dot(velocity, velocity))

# Diffusion tensor
Diff = fd.Identity(mesh.geometric_dimension())*(Dm + d_t*vnorm) + \
    fd.Constant(d_l-d_t)*fd.outer(velocity, velocity)/vnorm

# upwind term
vn = 0.5*(fd.dot(velocity, n) + abs(fd.dot(velocity, n)))

# transient term
F_t = phi*((c - c0) + K*c_mid)*fd.dx

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
          + phi*c_mid*vn*fd.ds
          - fd.inner(fd.grad(phi), velocity*c_mid)*fd.dx)

# full weak form
F = F_t + F_a + F_d

# %%
# 4) Solve problem
# -----------------
wave_speed = fd.conditional(fd.lt(np.abs(vnorm), tol), h_E/tol, h_E/vnorm)
# CFL
dt = 0.1*fd.interpolate(wave_speed, DG1).dat.data.min()
Dt.assign(dt)
outfile = fd.File("plots/adr_dg.pvd")

limiter = fd.VertexBasedLimiter(DG1)  # Kuzmin slope limiter

c_ = fd.Function(DG1, name="c")
problem = fd.LinearVariationalProblem(fd.lhs(F), fd.rhs(F), c_, bcs=bc)
solver = fd.LinearVariationalSolver(problem)

# initialize timestep
t = 0.0
it = 0

p = 0
while t < sim_time:
    # check dt
    dt = np.min([sim_time - t, dt])
    if (t + dt > ptimes[p]):
        dt = ptimes[p] - t
    Dt.assign(dt)

    # move next time step
    it += 1
    t += dt
    print("* iteration= {:4d}, dtime= {:8.6f}, time={:8.6f}".format(it, dt, t))

    # tracer
    solver.solve()

    # update sol.
    c0.assign(c_)
    limiter.apply(c0)

    # acochambramento c (c<1) and (c>0)
    # val = c0.dat.data
    # if np.any((val > 1.0) + (val < np.sqrt(tol))):
    #     val[val > 1.0] = 1.
    #     val[val < np.sqrt(tol)] = 0.
    #     c0.dat.data[:] = val

    # %%
    # 5) print results
    # if it % freq_res == 0:
    if t == ptimes[p]:
        if verbose:
            contours = fd.tripcolor(c0)
            plt.xlabel(r'$x$')
            plt.ylabel(r'$y$')
            plt.colorbar(contours)
            plt.show()

        c0.rename("c_h")
        outfile.write(c0, time=t)
        print('-- write results @ t= {}'.format(t))
        p += 1
        dt = 2e3


print('* normal termination.')
