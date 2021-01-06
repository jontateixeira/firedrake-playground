#!/home/john/opt/firedrake/bin/python3
# -*- coding: utf-8 -*-

"""
Solves the two phases flow through porous media problem in a full implicit
scheme.

Strong form:

    (lambda(s)*K)^(-1)*u + grad(p) = 0
                            div(u) = 0
          phi*ds/dt + u.grad(F(s)) = 0,

where,

    lambda(s) = (mu_rel*k_rw(s) + k_ro(s))/mu_w*mu_rel
         F(s) = k_rw(s)/mu_w/(k_rw(s)/mu_w + k_ro(s)/mu_o)
              = mu_rel*s^2/(s^2 + mu_rel*(1 - s)^2).

One can then can post-calculate the velocity of each phase using the
relation: u_j = - (k_rj(s)/mu_j)*K*grad(p).

Weak form:

Find u, p, s in W, such that,

   (v, (lambda*K)^(-1)*u) - (div(v), p) = - (v, pbar*n)_N       on Omega
                            (q, div(u)) = 0                     on Omega
        (r, phi*ds/dt) - (grad(r), F*u) = - (r, F*u.n)_N        on Omega

for all v, q, r in W'.

Model problem:

 -----4-----
 |         |
 1         2
 |         |
 -----3-----

Initial Conditions:
u(x, 0) = 0 in Omega
p(x, 0) = 0 in Omega
s(x, 0) = 0 in Omega

Boundary Conditions:
p(x, t) = pbar      on Gamma_2
u(x, t) = noflow    on Gamma_{3, 4} if u.n < 0
u(x, t) = qbar      on Gamma_1
s(x, t) = sbar      on Gamma_1 if u.n < 0
"""

# %%
# 0) importing modules
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
print('* loaded modules')

# %%
# 0.1) setting constants
EPS = np.finfo(np.float).eps
LEFT = 1
RIGHT = 2
BOTTOM = 3
TOP = 4
mD = 1e-13


# %%
# 1) Problem Parameters:
# ======================
print('* setting problem parameters')
oFile = "plots/two-phase.pvd"
# domain parameters
order = 1
n = 2
nx = 30
ny = 15
Lx = 60                             # m
Ly = 30                             # m
quad_mesh = True

# porous media
poro = 0.33
# permeability field generator parameter
k0 = 100       # reference perm mD
sigma = 2.5
par_a = 0.85
par_b = 2.0
par_c = 4.5

# viscosity
mu_w = 1e-3       # Pa.s
mu_rel = 20       # viscosity ratio (crude oil/water)

# boundary conditions parameters
qbar = (1.0 / 86400, 0.0)    # m/day
inlet = LEFT

pbar = 10001.         # Pa
outlet = RIGHT

q0bar = (0.0, 0.0)   # m/day
noflow = [TOP, BOTTOM]

sbar = 1.0

# problem parameters
verbose = True
freq_res = 50
nsteps = 1000
sim_time = 864000   # s


# %%
# Process section (simulation)
# ============================

# 2) Define mesh
print('* define mesh')
mesh = fd.RectangleMesh(nx * n, ny * n, Lx, Ly, quadrilateral=quad_mesh)

if verbose:
    fd.triplot(mesh)
    plt.legend()
    plt.savefig("plots/twophase.png")

# ----
# 3) Setting problem (FunctionSpace, Init.Condition, VariationalForms)
print('* setting problem')

# 3.1) # Define function space for system
if quad_mesh:
    RT1 = fd.FunctionSpace(mesh, "RTCF", order)
    DG0 = fd.FunctionSpace(mesh, "DQ", order - 1)

    # Others function space
    V = fd.VectorFunctionSpace(mesh, "DQ", order - 1)  # pre-process purporse
    T = fd.TensorFunctionSpace(mesh, "DQ", order - 1)  # kinv
else:
    RT1 = fd.FunctionSpace(mesh, "RT", order)
    DG0 = fd.FunctionSpace(mesh, "DG", order - 1)

    # Others function space
    V = fd.VectorFunctionSpace(mesh, "DG", order - 1)  # pre-process purporse
    T = fd.TensorFunctionSpace(mesh, "DG", order - 1)  # kinv

W = fd.MixedFunctionSpace([RT1, DG0, DG0])
P1 = fd.VectorFunctionSpace(mesh, "CG", order)     # post-process purporse


# test and trial functions on the subspaces of the mixed function spaces as
# follows: ::
v, q, r = fd.TestFunctions(W)


# %%
# 3.2) material property
# perm lognormal
Kinv = fd.Function(T, name="Kinv")

k = np.random.randn(nx * n, ny * n) + par_a*np.random.randn(1)
kf = ndimage.gaussian_filter(k, sigma)
kl = k0*np.exp(par_b+par_c*kf) * mD


if verbose:
    fig, axes = plt.subplots(ncols=3)

    img0 = axes[0].imshow(k, interpolation='nearest',
                          cmap='viridis', origin='lower')
    axes[0].set(title='original random data')
    fig.colorbar(img0, ax=axes[0])

    img1 = axes[1].imshow(kf, interpolation='nearest',
                          cmap='viridis', origin='lower')
    axes[1].set(title='smoothed data')
    fig.colorbar(img1, ax=axes[1])

    img2 = axes[2].imshow(kl/mD, interpolation='nearest',
                          cmap='viridis', origin='lower')
    axes[2].set(title='lognormal data')
    fig.colorbar(img2, ax=axes[2])
    plt.savefig('plots/twophase_perm.png')

# assign correctly permeability values
ccenter = fd.interpolate(mesh.coordinates, V)


@np.vectorize
def fix_perm_map(x, y):
    i = np.floor(x / (Lx / (nx * n))).astype(int)
    j = np.floor(y / (Ly / (ny * n))).astype(int)
    return kl[i, j]


Kinv.dat.data[:, 0, 0] = 1 / \
    fix_perm_map(ccenter.dat.data[:, 0], ccenter.dat.data[:, 1])
Kinv.dat.data[:, 1, 1] = 1 / \
    fix_perm_map(ccenter.dat.data[:, 0], ccenter.dat.data[:, 1])


def lmbdainv(s):
    # Total mobility
    return mu_rel*mu_w/(mu_rel*s**2 + (1.0 - s)**2)


def Fw(s):
    # Fractional flow function
    return mu_rel*s**2/(mu_rel*s**2 + (1.0 - s)**2)


# ---
# 3.3) initial cond.
U0 = fd.Function(W)
u0, p0, s0 = fd.split(U0)
U = fd.Function(W)
u, p, s = fd.split(U)


# ----
# set boundary conditions
# The strongly enforced boundary conditions on the BDM space on the top and
# bottom of the domain are declared as: ::
bc0 = fd.DirichletBC(W.sub(0), fd.Constant(qbar), inlet)
bc1 = fd.DirichletBC(W.sub(0), fd.Constant(q0bar), noflow)
bc3 = fd.DirichletBC(W.sub(2), fd.Constant(sbar), inlet, method='geometric')


# -------
# 3.4) Variational Form

# Time step
dt = sim_time / nsteps
dtc = fd.Constant(dt)

# saturation
s_mid = 0.5*(s0 + s)
s_in = fd.Constant(sbar)

# source term
f = fd.Constant(0.0)

# normal face
n = fd.FacetNormal(mesh)

phi = fd.Constant(poro)

# the bilinear and linear forms of the variational problem are defined as: ::
F_p = fd.dot(v, lmbdainv(s_mid)*Kinv*u)*fd.dx - \
    fd.div(v)*p*fd.dx + q*fd.div(u)*fd.dx \
    - q*f*fd.dx + fd.Constant(pbar)*fd.inner(v, n)*fd.ds(outlet)

# Upwind normal velocity: (dot(vD, n) + |dot(vD, n)|)/2.0
un = 0.5*(fd.dot(u0, n) + abs(fd.dot(u0, n)))

F_s = r*phi*(s - s0)*fd.dx - \
    dtc*(fd.dot(fd.grad(r), Fw(s_mid)*u)*fd.dx - r*Fw(s_mid)*un*fd.ds
         - fd.conditional(fd.dot(u0, n) < 0, r *
                          fd.dot(u0, n)*Fw(s_in), 0.0)*fd.ds
         - (r('+') - r('-'))*(un('+')*Fw(s_mid)('+') -
                              un('-')*Fw(s_mid)('-'))*fd.dS)

# Res.
R = F_p + F_s

prob = fd.NonlinearVariationalProblem(R, U, bcs=[bc0, bc1, bc3])
solver2ph = fd.NonlinearVariationalSolver(prob)


# %%
# 4) Solve problem
# initialize variables
xv, xp, xs = U.split()
vel = fd.project(u0, P1)
xp.rename('pressure')
xs.rename('saturation')
vel.rename('velocity')


outfile = fd.File(oFile)
it = 0
t = 0.0
while t < sim_time:
    t += dt
    print("*iteration= {:3d}, dtime= {:8.6f}, time={:8.6f}".format(it, dt, t))

    solver2ph.solve()

    # update solution
    U0.assign(U)

    # 5) print results
    if it % freq_res == 0 or t == sim_time:
        vel.assign(fd.project(U.sub(0), P1))
        xp.assign(U.sub(1))
        xs.assign(U.sub(2))
        outfile.write(vel, xp, xs, time=t)
        print('*write results in: {:s}'.format(oFile))

    # move next time step
    it += 1
print('*normal termination.')
