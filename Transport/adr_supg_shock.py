#!/home/john/opt/firedrake/bin/python3
# -*- coding: utf-8 -*-

"""
Solves the unsteady state advection-diffusion-reaction problem, using SUPG and
shock capturing term

Strong form (SF):

          dc/dt + div(c*u) = div(D*grad(c)) - k*c^n + f

The problem is either advection- or diffusion-dominated, depending on the ratio
u*h/D, where h is the characteristic length scale.

Weak form:

Find c in W, such that,

        (w, dc/dt) + (w, u.grad(c)) + (grad(w),D*grad(c)) 
                   + (w, k*c^n) + SUPG + Shock = (w, f)            on Omega

where,
        SUPG  = (grad(w),tau*res*u)
          res = dc/dt + (w, u.grad(c)) + (grad(w),D*grad(c)) 
                   + (w, k*c^n) - (w, f)
          tau = h_mesh/(2*||u||)
        Shock = (grad(w),vshock*|res|*grad(c))
                 { Beta*h_mesh*abs(res)/(2*||grad(c)||), if ||grad(c)|| > 0
       vshock <--{
                 { 0, o.w

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
print('* loaded modules')

# %%
# 0.1) setting constants
LEFT = 1
RIGHT = 2
BOTTOM = 3
TOP = 4

# %%
# 1) Problem Parameters:
# ======================
print('* setting problem parameters')

# domain parameters
order = 1
n = 4
nx = 30
ny = 5
Lx = 60                             # m
Ly = 10                             # m
quad_mesh = True


# reaction/transport parameter
Diff = 1e-1                 # diffusion coefficient
K = 0.01                      # reaction rate
h_n = fd.Constant(0.)       # outflow condition (BC)
cIn = 1.0
s = 0.0
add_shock_term = True
beta = 1.0  # 2.0

# problem parameters
verbose = False
freq_res = 50
CFL = 0.25
sim_time = 50.0     # simulation time

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
    plt.savefig("plots/adr_supg_mesh.png")

# 3) Setting problem (FunctionSpace, Init.Condition, VariationalForms)

# 3.1) # Define function space for system of concentrations (transport)
X = fd.FunctionSpace(mesh, "CG", order)
V = fd.VectorFunctionSpace(mesh, "CG", order + 1)

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
vel = fd.Function(V, name='velocity').interpolate(fd.as_vector([1.0, 0.]))
K = fd.Constant(K)
Diff = fd.Constant(Diff)
Dt = fd.Constant(1e-3)
c_mid = 0.5 * (c + c0)  # Crank-Nicolson timestepping
fc = fd.Constant(s)

# weak form (transport)
F1 = w*(c - c0)*fd.dx + Dt*(w*fd.dot(vel, fd.grad(c_mid))
                            + Diff*fd.dot(fd.grad(w),
                                          fd.grad(c_mid)) + w*K*c_mid
                            - fc*w)*fd.dx  # - Dt*h_n*w*fd.ds(outlet)

# strong form
R = (c - c0) + Dt*(fd.dot(vel, fd.grad(c_mid)) -
                   Diff*fd.div(fd.grad(c_mid)) + K*c_mid - fc)


# *** Adding SUPG stabilizing and shock cap. terms ***
# SUPG stabilisation parameters
vnorm = fd.sqrt(fd.dot(vel, vel))
h = fd.sqrt(2)*fd.CellVolume(mesh) / fd.CellDiameter(mesh)
# h = fd.CellSize(mesh)
tau = h / (2.0 * vnorm)
# tau = fd.pow(1/(0.5*Dt) + 2.0*vnorm/h + 4*Diff/fd.pow(h, 2.0), -1)

# Residual and stabilizing terms
F1 += tau*fd.dot(vel, fd.grad(w)) * R * fd.dx

# shock capturing parameters
if add_shock_term:
    cnorm = fd.sqrt(fd.dot(fd.grad(c0), fd.grad(c0)))
    vshock = fd.conditional(fd.gt(cnorm, 1e-15),
                            beta*h / (2*cnorm),  # *cnorm,
                            fd.Constant(0.))

    # shock terms
    # F1 += vshock*fd.dot(fd.grad(w), fd.grad(c_mid))*fd.dx
    F1 += vshock*np.abs(R)*fd.inner(fd.grad(w), fd.grad(c_mid))*fd.dx

F = F1


t_params = {
    'snes_type': 'newtonls',
    'snes_max_it': 100,
    'ksp_type': 'gmres',
    'pc_type': 'sor',
    'ksp_rtol': 1e-6,
    'ksp_max_it': 1000
}
# 'pc_type': 'bjacobi',
# 'pc_type': 'sor',
# 'snes_monitor': True,
# 'ksp_monitor': True,
# 'ksp_type': 'lgmres',
# 'pc_type': 'ilu',
# 'mat_type': 'aij',
# 'ksp_rtol': 1e-8,
# 'ksp_max_it': 2000,
# 'ksp_monitor_true_residual': None
#

prob = fd.NonlinearVariationalProblem(F, c, bcs=t_bc)
transport = fd.NonlinearVariationalSolver(prob, solver_parameters=t_params)

# %%
# 4) Solve problem
c0.assign(0.)
t = it = 0
outfile = fd.File("plots/adr_supg.pvd")
c0.rename("c")

dt = CFL*fd.interpolate(h/vnorm, DG0).dat.data.min()
Dt.assign(dt)
dt0 = dt

Pe = vnorm * h / (2.0 * Diff)
print('Peclet = {}; dt = {}'.format(fd.interpolate(Pe, DG0).dat.data.mean(), dt))
D = fd.interpolate(vshock*np.abs(R), DG0)
D.rename("visc_shock")

while t < sim_time:
    # move next time step
    t += np.min([sim_time-t, dt])
    print("* iteration= {:4d}, dtime= {:8.6f}, time={:8.6f}".format(it, dt, t))

    Pe = vnorm * h / (2.0 * Diff)
    print('Peclet = {}'.format(fd.interpolate(Pe, DG0).dat.data.mean()))

    # transport
    transport.solve()
    it += 1
    # res = fd.norm(fd.interpolate(R, DG0))
    # print('Residual = {}'.format(res))

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
        if add_shock_term:
            D.assign(fd.interpolate(vshock*np.abs(R), DG0))
            outfile.write(vel, c0, D, time=t)
        else:
            outfile.write(vel, c0, time=t)
