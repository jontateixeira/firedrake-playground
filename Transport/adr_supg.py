#!/home/john/opt/firedrake/bin/python3
# -*- coding: utf-8 -*-
"""
Solves the unsteady state advection-diffusion-reaction problem, using SUPG

Strong form (SF):

          dc/dt + div(c*u) = div(D*grad(c)) - k*c^n + f

The problem is either advection- or diffusion-dominated, depending on the ratio
u*h/D, where h is the characteristic length scale.

Weak form:

Find c in W, such that,

        (w, dc/dt) + (w, u.grad(c)) + (grad(w),D*grad(c)) 
                   + (w, k*c^n) + SUPG = (w, f)            on Omega

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
c(x, 0) = 0 in Omega

Boundary Conditions:
c(x, t) = cbar      on Gamma_1
"""

# %%
# 0) importing modules
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt
print('* modules loaded')

# %%
# 0.1) setting constants
LEFT = 1
RIGHT = 2
BOTTOM = 3
TOP = 4

# %%
# 1) Problem Parameters:
# ======================
print('* problem setting')

# domain parameters
order = 1
n = 2
nx = 30
ny = 5
Lx = 1.0                             # m
Ly = 0.4                             # m
quad_mesh = True


# reaction/transport parameter
# reaction/transport parameter
Dm = 1e-9                   # diffusion (m²/s)
d_l = 0.0                   # longitidinal dispersion (m)
d_t = 0.0                   # transversal dispersion (m)
K = 0.00                    # reaction rate (mol/m³/s)
s = 0.0                     # source

# boundary conditions parameters
inlet = LEFT
v_inlet = fd.Constant((1.0e-6, .0))
cIn = 1.0                   # injection conc

# problem parameters
verbose = False
freq_res = 10
dt = 2000          # time steps size (initial)
sim_time = 1.0e6     # simulation time

ptimes = [2e5, 4e5, 6e5, 8e5, 1e6]


# %%
# Process section (simulation)
# ============================

# 2) Define mesh
print('* define mesh')
mesh = fd.RectangleMesh(nx * n, ny * n, Lx, Ly, quadrilateral=quad_mesh)

if verbose:
    fd.triplot(mesh)
    plt.legend()
    plt.savefig("plots/adr_supg_mesh.png")

# 3) Setting problem (FunctionSpace, Init.Bound.Condition, VariationalForms)

# 3.1) # Define function space for system of concentrations (transport)
X = fd.FunctionSpace(mesh, "CG", order)
V = fd.VectorFunctionSpace(mesh, "CG", order)

# Others function space
DG0 = fd.FunctionSpace(mesh, "DG", 0)

# 3.1) Define trial and test functions
w = fd.TestFunction(X)


# 3.2) Set initial conditions
c0 = fd.Function(X, name="c")
c = fd.Function(X)


# 3.3) set boundary conditions
t_bc = fd.DirichletBC(X, cIn, inlet)


# ======================
# 3.4) Variational Form
# coefficients
# advective velocity
vel = fd.Function(V, name='velocity')
vel.interpolate(v_inlet)
vnorm = fd.sqrt(fd.dot(vel, vel))


# Diffusion tensor
Diff = fd.Identity(mesh.geometric_dimension())*(Dm + d_t*vnorm) + \
    fd.Constant(d_l-d_t)*fd.outer(vel, vel)/vnorm
Diff = fd.Identity(mesh.geometric_dimension())*Dm

Dt = fd.Constant(dt)
K = fd.Constant(K)
c_mid = 0.5 * (c + c0)  # Crank-Nicolson timestepping
fc = fd.Constant(s)

# weak form (transport)
F = w*(c - c0)*fd.dx + Dt*(w*fd.dot(vel, fd.grad(c_mid))
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
# tau = pow(1/(0.5*Dt) + 2.0*vnorm/h + 4*Diff/pow(h, 2.0), -1)

# Residual and stabilizing terms
F += tau*fd.dot(vel, fd.grad(w)) * R * fd.dx

c0.assign(0.)
outfile = fd.File("plots/adr_supg.pvd")


dt = 0.07*fd.interpolate(h/vnorm, DG0).dat.data.min()
dt0 = dt

problem = fd.NonlinearVariationalProblem(F, c, bcs=t_bc)
transport = fd.NonlinearVariationalSolver(problem)

# %%
# 4) Solve problem
t = 0.0
it = 0


p = 0
while t < sim_time:
    # check dt
    dt = np.min([sim_time-t, dt]) if (sim_time-t) > dt else dt0
    if (t + dt > ptimes[p]):
        dt = ptimes[p] - t
    # move next time step
    Dt.assign(dt)

    # move next time step
    it += 1
    t += dt
    print("* iteration= {:4d}, dtime= {:8.6f}, time={:8.6f}".format(it, dt, t))


    Pe = vnorm * h / (2.0 * fd.det(Diff))
    print('Peclet = {}'.format(fd.interpolate(Pe, X).dat.data.mean()))

    # transport
    transport.solve()
    
    res = fd.norm(fd.interpolate(R, X))
    print('Residual = {}'.format(res))

    # update sol.
    c0.assign(c)

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

        outfile.write(vel, c0, time=t)
        p += 1
        dt = 2e3


print('* normal termination.')
