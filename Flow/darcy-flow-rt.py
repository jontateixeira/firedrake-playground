# -*- coding: utf-8 -*-
"""
Solves pressure-driven, steady-state Darcy's flow on a square plate with 
spatially varying permeability (inspired in MRST logperm function).

        Kinv*u + grad(p) = 0
                  div(u) = 0
        
                       p = p_0  in Gamma_D
                     u*n = g_0  in Gamma_N

With the weak form:

 (v, Kinv*u) - (div(v), p) = - (v, p*n)_N  for all v
               (q, div(u)) = 0             for all q
"""

# %%
# 0) importing modules
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from ufl.constant import Constant
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
ny = 5
Lx = 10                             # m
Ly = 10                             # m

# permeability field generator parameter
sigma = 2.5
par_a = 0.85
par_b = 2.0
par_c = 4.5

# boundary conditions parameters
qbar = (1.0, 0.0)
inlet = LEFT

pbar = 0.001
outlet = RIGHT

q0bar = (0.0, 0.0)
noflow = [TOP, BOTTOM]

# problem parameters
verbose = True
freq_res = 50
CFL = 0.25
sim_time = 50.0     # simulation time

# %%
# Process section (simulation)
# ============================

# 2) Define mesh
print('* define mesh')
mesh = fd.RectangleMesh(nx * n, ny * n, Lx, Ly)

if verbose:
    fd.triplot(mesh)
    plt.legend()
    plt.savefig("plots/darcy_rt.png")

# ----
# 3) Setting problem (FunctionSpace, Init.Condition, VariationalForms)
print('* setting problem')

# 3.1) # Define function space for system
RT = fd.FunctionSpace(mesh, "RT", order)
DG = fd.FunctionSpace(mesh, "DG", order - 1)
W = RT * DG

# Others function space
V = fd.VectorFunctionSpace(mesh, "DG", order - 1)
T = fd.TensorFunctionSpace(mesh, "DG", order - 1)

# test and trial functions on the subspaces of the mixed function spaces as
# follows: ::
u, p = fd.TrialFunctions(W)
v, q = fd.TestFunctions(W)

# -----
# 3.2) material property

# perm lognormal
Kinv = fd.Function(T, name="Kinv")


k = np.random.randn(nx * n, ny * n) + par_a*np.random.randn(1)
kf = ndimage.gaussian_filter(k, sigma)
kl = np.exp(par_b+par_c*kf)


if verbose:
    fig, axes = plt.subplots(ncols=3)

    img0 = axes[0].imshow(k, interpolation='nearest', cmap='viridis')
    axes[0].set(title='original random data')
    fig.colorbar(img0, ax=axes[0])

    img1 = axes[1].imshow(kf, interpolation='nearest', cmap='viridis')
    axes[1].set(title='smoothed data')
    fig.colorbar(img1, ax=axes[1])

    img2 = axes[2].imshow(kl, interpolation='nearest', cmap='viridis')
    axes[2].set(title='lognormal data')
    fig.colorbar(img2, ax=axes[2])
    plt.savefig('plots/perm_gen.png')

# assign correctly permeability values
ccenter = fd.interpolate(mesh.coordinates, V)


@np.vectorize
def fix_perm_map(x, y):
    i = np.floor(x / (Lx / (nx * n))).astype(int)
    j = np.floor(y / (Ly / (ny * n))).astype(int)
    return kl[j, i]


Kinv.dat.data[:, 0, 0] = 1 / \
    fix_perm_map(ccenter.dat.data[:, 0], ccenter.dat.data[:, 1])
Kinv.dat.data[:, 1, 1] = 1 / \
    fix_perm_map(ccenter.dat.data[:, 0], ccenter.dat.data[:, 1])

# -------
# 3.4) Variational Form
# the bilinear and linear forms of the variational problem are defined as: ::
a = fd.dot(v, Kinv*u)*fd.dx - fd.div(v)*p*fd.dx + q*fd.div(u)*fd.dx

f = fd.Constant(0.0)
n = fd.FacetNormal(mesh)
L = q*f*fd.dx - fd.Constant(pbar)*fd.inner(v, n)*fd.ds(outlet)

# ----
# 3.5) set boundary conditions
# The strongly enforced boundary conditions on the BDM space on the top and
# bottom of the domain are declared as: ::
bc0 = fd.DirichletBC(W.sub(0), fd.Constant(qbar), inlet)
bc1 = fd.DirichletBC(W.sub(0), fd.Constant(q0bar), noflow)

# ----
# 4) Define and solve the problem
#
# Now we're ready to solve the variational problem. We define `w` to be a
# function to hold the solution on the mixed space.
w = fd.Function(W)
# problem = fd.LinearVariationalProblem(a, L, w, bcs=[bc0, bc1])

# solve
fd.solve(a == L, w, bcs=[bc0, bc1])


# %%
# Post-process and plot interesting fields
print('* post-process')

# collect fields
vel, press = w.split()
press.rename('pressure')

# Projecting velocity field to a continuous space (visualization purporse)
P1 = fd.VectorFunctionSpace(mesh, "CG", 1)
vel_proj = fd.project(vel, P1)
vel_proj.rename('velocity_proj')

# project permeability Kxx
kxx = fd.Function(DG)
kxx.rename('Kxx')
kxx.dat.data[...] = 1 / Kinv.dat.data[:, 0, 0]

# print results
fd.File("plots/darcy_rt.pvd").write(vel_proj, press, kxx)
