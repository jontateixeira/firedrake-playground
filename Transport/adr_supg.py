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
n = 4
nx = 30
ny = 5
Lx = 60                             # m
Ly = 10                             # m
quad_mesh = True


# reaction/transport parameter
Dif = 1e-2                  # diffusion (m²/s)
d_l = 4.0                   # longitidinal dispersion (m)
d_t = 1.0                   # transversal dispersion (m)
K = 0.00                    # reaction rate (mol/m³/s)
h_n = fd.Constant(0.)       # outflow condition (BC)
cIn = 1.0                   # injection conc
fcenter = 2
s = 0.0                     # source

# problem parameters
verbose = False
freq_res = 10
CFL = 0.5          # CFL number
dt = 0.5          # time steps size (initial)
sim_time = 40.0     # simulation time

# boundary conditions parameters
inlet = LEFT
outlet = RIGHT


# %%
# Process section (simulation)
# ============================

# 2) Define mesh
print('* define mesh')
mesh = fd.RectangleMesh(nx * n, ny * n, Lx, Ly, quadrilateral=quad_mesh)

if verbose:
    fd.triplot(mesh)
    plt.legend()
    plt.savefig("plots/SB_mesh.png")

# 3) Setting problem (FunctionSpace, Init.Bound.Condition, VariationalForms)

# 3.1) # Define function space for system of concentrations (transport)
X = fd.FunctionSpace(mesh, "CG", order)
V = fd.VectorFunctionSpace(mesh, "CG", order)

# Others function space
DG0 = fd.FunctionSpace(mesh, "DG", 0)

# 3.1) Define trial and test functions
w = fd.TestFunction(X)


# 3.2) Set initial conditions
c0 = fd.Function(X, name="conc0")
c = fd.Function(X, name="conc")


# 3.3) set boundary conditions
t_bc = fd.DirichletBC(X, cIn, inlet)


# ======================
# 3.4) Variational Form
# coefficients
# advective velocity
vel = fd.Function(V, name='velocity').interpolate(fd.as_vector([1.000, 0.]))
vnorm = fd.sqrt(fd.dot(vel, vel))
K = fd.Constant(K)
# Diffusion tensor
Diff = fd.Identity(mesh.geometric_dimension())*(Dif + d_t*vnorm) + \
    fd.Constant(d_l-d_t)*fd.outer(vel, vel)/vnorm
# print('Diffusion = {}'.format(fd.interpolate(fd.det(Diff), DG0).dat.data.mean()))

Dt = fd.Constant(dt)
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


prob = fd.NonlinearVariationalProblem(F, c, bcs=t_bc)
transport = fd.NonlinearVariationalSolver(prob)

# %%
# 4) Solve problem
c0.assign(0.)
t = it = 0
dt0 = dt
outfile = fd.File("plots/adr_supg.pvd")
c0.rename("c")
while t < sim_time:
    # move next time step
    t += np.min([sim_time-t, dt])
    print("* iteration= {:4d}, dtime= {:8.6f}, time={:8.6f}".format(it, dt, t))

    Pe = vnorm * h / (2.0 * fd.det(Diff))
    print('Peclet = {}'.format(fd.interpolate(Pe, DG0).dat.data.mean()))

    # transport
    Dt.assign(np.min([sim_time-t, dt]))
    transport.solve()
    it += 1
    res = fd.norm(fd.interpolate(R, DG0))
    print('Residual = {}'.format(res))

    dt = CFL*fd.interpolate(h/vnorm, DG0).dat.data.min()
    dt = np.min([sim_time-t, dt]) if (sim_time-t) > dt else dt0

    # update sol.
    c0.assign(c)

    # %%
    # 5) print results
    if it % freq_res == 0:
        if verbose:
            contours = fd.tripcolor(c0)
            plt.xlabel(r'$x$')
            plt.ylabel(r'$y$')
            plt.colorbar(contours)
            plt.show()

        outfile.write(vel, c0, time=t)
