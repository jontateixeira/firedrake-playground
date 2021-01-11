# -*- coding: utf-8 -*-
"""
Solves steady-state Stokes-Brinkman flow on a square plate.

   -div(2*mu*D(u) - p*Id) + mu/k*u = f    in Omega
                            div(u) = g    in Omega
             -(2*mu*D(u) - p*Id).n = gN   in Gamma_N
                                 u = gD   in Gamma_D

where,

      D(u) = 0.5*(grad(u) + grad(u).T)
      Id   = Identity matrix

With the weak form:

(2*mu*D(v), D(u)) - (v, div(u))
                  + (mu/k*u,v) =  (f, v) + (gN, v)_N  for all v
                   (q, div(u)) = 0                    for all q
"""

# %%
# 0) importing modules
import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
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
nx = 5
ny = 10
Lx = 60                             # m
Ly = 30                             # m
quad_mesh = True

# material property
# fluid parameters
rho = 1000.                 # fluid density [kg/m3]
mu = 1.0e-3                 # fluid viscosity [N.s/m2 ~ Pa.s]
mueff = [mu, mu]            # effective viscosity


# rock parameters
k_cavern, k_porous = 1e+3, 1e-13     # permeability [m2]
wx, wy = 0.25, 0.25


# boundary conditions
inlet = LEFT
outlet = RIGHT
noslip = [TOP, BOTTOM]

v_noslip = fd.Constant((.0, .0))
v_inlet = fd.Constant((0.100, .0))
p_out = 1e5                         # Pressure [Pa]
# g_n = fd.Constant(-2e5)
f = fd.Constant((0., rho*0.))

# problem parameters
verbose = False
Beta = - 1              # [+1,-1] non-symmetric/symmetric problem
gama0 = 100             # stabilization parameter (sufficient large)
k = 3

# %%
# Process simulation
# ------------------

# 2) Define mesh
print('* define mesh')
mesh = fd.RectangleMesh(nx * n, ny * n, Lx, Ly, quadrilateral=quad_mesh)

if verbose:
    fd.triplot(mesh)
    plt.legend()
    plt.savefig("SB_hdiv_mesh.png")

# Define subdomains
x = fd.SpatialCoordinate(mesh)
domain = fd.And(fd.And(x[0] > Lx*wx, x[1] > Ly*wy),
                fd.And(x[0] < Lx*(1-wx), x[1] < Ly*(1-wy)))
square = fd.conditional(domain, 1/k_cavern, 1/k_porous)


ikm = fd.Function(fd.FunctionSpace(mesh, "DG", 0), name='invK')
ikm.interpolate(square)
if verbose:
    contours = fd.tripcolor(ikm, cmap="viridis")
    plt.colorbar(contours)
    plt.savefig("plots/SB_hdiv_doamins.png")


# 3) Setting problem (FunctionSpace, Init.Condition, VariationalForms)
# 3.1) Function spaces
if quad_mesh:
    RT1 = fd.FunctionSpace(mesh, "RTCF", order)
    DG0 = fd.FunctionSpace(mesh, "DQ", order - 1)
else:
    RT1 = fd.FunctionSpace(mesh, "RT", order)
    DG0 = fd.FunctionSpace(mesh, "DG", order - 1)

# Create mixed function spaces
W = RT1 * DG0

# 3.1) Define trial and test functions
# SB
u, p = fd.TrialFunctions(W)
v, q = fd.TestFunctions(W)


# ----------------------
# 3.2) Variational Form
# ----------------------
# SB


# coefficients
mu = fd.Constant(mu)
rho = fd.Constant(rho)

n = fd.FacetNormal(mesh)
mul = fd.Function(DG0)
mul.interpolate(fd.conditional(domain, mueff[0], mueff[1]))


def D(x):
    return 2*fd.sym(fd.grad(x))


# variational form (SB)
a = 2*mu*fd.inner(D(v), D(u))*fd.dx - p*fd.div(v)*fd.dx + \
    mu*ikm*fd.inner(u, v)*fd.dx + q*fd.div(u)*fd.dx

L = fd.inner(f, v)*fd.dx + p_out*fd.inner(v, n)*fd.ds(outlet)

# tangential components
t = fd.as_vector((-n[1], n[0]))
gamma = fd.Constant(gama0*k/(Lx/nx/4*Ly/ny/4))

a = a - 2*mu*fd.inner(fd.avg(D(u)*n), fd.jump(v))*fd.dS -\
    2*mu*Beta*fd.inner(fd.avg(D(v)*n), fd.jump(u))*fd.dS


# 3.3) set boundary conditions
# SB
bcs = [fd.DirichletBC(W.sub(0), v_noslip, noslip),
       fd.DirichletBC(W.sub(0), v_inlet, inlet)]

# ----
# 4) Define and solve the problem
#
print('* solving...')
# Now we're ready to solve the variational problem. We define `w` to be a
# function to hold the solution on the mixed space.
s = fd.Function(W)

# solve
fd.solve(a == L, s, bcs=bcs)


# %%
# Post-process and plot interesting fields
print('* post-process')

# collect fields
vel, press = s.split()
press.rename('pressure')

# Projecting velocity field to a continuous space (visualization purporse)
P1 = fd.FunctionSpace(mesh, "CG", 1)
vP1 = fd.VectorFunctionSpace(mesh, "CG", 1)
vel_proj = fd.interpolate(vel, vP1)
vel_proj.rename('velocity_H1')

# print results
fd.File("plots/sb_rt.pvd").write(vel_proj,
                                 fd.interpolate(press, P1),
                                 ikm)

# fig, axes = plt.subplots(nrows=2)
# pcor = fd.tripcolor(press, axes=axes[0], shading='flat', cmap='jet')
# cbar = fig.colorbar(pcor, ax=axes[0])
# cbar.set_label('Pressure [Pa]')
# # axes[0].set_title('pressure solution')
# axes[0].set_aspect('equal')
# axes[0].axis('off')

# pcor = fd.tripcolor(vel_proj, axes=axes[1], shading='flat', cmap='jet')
# cbar = fig.colorbar(pcor, ax=axes[1])
# cbar.set_label('Velocity mag [m/s]')
# # axes[1].set_title('solution')
# axes[1].set_aspect('equal')
# axes[1].axis('off')
# plt.show()


print('* normal termination')
