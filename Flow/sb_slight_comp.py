#!/home/john/opt/firedrake/bin/python3
# -*- coding: utf-8 -*-
"""
Solves steady-state Stokes-Brinkman flow on a square plate.

   -div(2*mu*D(u) - p*Id) + mu/k*u = f    in Omega
            phi*c_f*dp/dt - div(u) = g    in Omega
             -(2*mu*D(u) - p*Id).n = gN   in Gamma_N
                                 u = gD   in Gamma_D

where,

      D(u) = 0.5*(grad(u) + grad(u).T)
      Id   = Identity matrix

With the weak form:

(2*mu*D(v), D(u)) - (v, div(u))
                    + (mu/k*u,v) =  (f, v) + (gN, v)_N  for all v
(q, phi*c_f*dp/dt) - (q, div(u)) = 0                    for all q
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
Lx = 60                             # m
Ly = 10                             # m
quad_mesh = True


# materials parameters
# fluid parameters
rho = 1000.                 # fluid density [kg/m3]
mu = 1.0e-3                 # fluid viscosity [N.s/m2 ~ Pa.s]
mueff = [mu, mu]            # effective viscosity
c_t = 1e-9                  # total compressibility [1/Pa]


# SB parameters
k_cavern, k_porous = 1e-13, 1e-13    # permeability [m2]
phi = [0.1, 0.1]                     # effective porosity [-]
wx, wy = 0.0, 0.25
f = fd.Constant((0., rho*0.))

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

# Others function space
DG0 = fd.FunctionSpace(mesh, "DG", 0)

# Create mixed function spaces
W = V*P
# W = fd.MixedFunctionSpace([V, P])

# 3.1) Define trial and test functions
# SB
u, p = fd.TrialFunctions(W)
v, q = fd.TestFunctions(W)


# 3.2) Set initial conditions
# ---- previous solution
# SB
w0 = fd.Function(W, name="pv0")
(vD0, p0) = w0.split()


# 3.3) set boundary conditions
# SB
bcs = [fd.DirichletBC(W.sub(0), v_noslip, noslip),
       fd.DirichletBC(W.sub(1), p_out*10, inlet),
       fd.DirichletBC(W.sub(1), p_out, outlet)]

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

# mesh size
h = fd.sqrt(2)*fd.CellVolume(mesh) / fd.CellDiameter(mesh)
# advective velocity
vel = fd.Function(V, name='velocity')
vnorm = fd.sqrt(fd.dot(vel, vel))


# %%
# 4) Solve problem
# -----------------
t = 0.0
it = 0
dt0 = dt
cfl_conditional = fd.conditional(fd.lt(np.abs(vnorm), tol), 1/tol, h/vnorm)
outfile = fd.File("plots/sb_compress.pvd")

sol = fd.Function(W)

# initialize timestep
while t < sim_time:
    # move next time step
    print("* iteration= {:4d}, dtime= {:8.6f}, time={:8.6f}".format(it, dt, t))

    # SB
    idt.assign(1 / dt)
    fd.solve(a == L, sol, bcs=bcs)

    vel.assign(sol.sub(0))
    numCFL = dt/fd.interpolate(cfl_conditional, DG0).dat.data.min()
    print('CFL = {}; vel. = {}'.format(numCFL, np.sqrt(np.sum(
        vel.dat.data*vel.dat.data, axis=1)).max()))

    dt = CFL*fd.interpolate(cfl_conditional, DG0).dat.data.min()
    dt = np.min([sim_time-t, dt]) if (sim_time-t) > dt else dt0

    p0.assign(sol.sub(1))

    # %%
    # 5) print results
    if it % freq_res == 0:

        v, p = sol.split()
        p.rename("pressure")
        v.rename("velocidade")
        outfile.write(p, v, time=t)

    # move next time step
    t += dt
    it += 1

print('* normal termination.')
