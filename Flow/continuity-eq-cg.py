# -*- coding: utf-8 -*-
"""
Solves continuity, compressible hydrostatic equation for porous media.

        d(rho) 
        ------ + div(rho*v) = 0
          dt
        
           p = p_0  in Gamma_D
         u*n = g_0  in Gamma_N

With the weak form:

a(p,q) = (rho(p), q) - dt*(rho(p)*K/mu*grad(p), grad(q)) 
         - dt*(div(rho(p)^2*K/mu*g),q)

  L(q) = dt*(f,q) + dt*(g_0,q)_N + (rho(p), q)
"""
# %%
# importing modules
import firedrake as fd
import numpy as np

# %%
# setting constants
LEFT = 1
RIGHT = 2
BOTTOM = 3
TOP = 4



# %%
# mesh parameters
# ---------------
verbose = False
n = 50
quad_mesh = True

# define mesh
mesh = fd.UnitSquareMesh(n, n, quadrilateral=quad_mesh)

# get usefull mesh info
dim = mesh.geometric_dimension()
h_mesh = fd.Constant(1.0 / n)
x, y = fd.SpatialCoordinate(mesh)

# %%
# Boundary condition
# ------------------
# -> essential 
prd_well = fd.conditional((x - 1)**2 + (y - 1)**2 < h_mesh**2, \
                          1e5, 0.) # Pa

# -----
# sink
# way #1: natural bdr condition
# src = None
# nat_bdr = fd.conditional(x**2 + y**2 < h_mesh**2,
#                          -0.1, 0.) # 10 m3/d

# way #2: source term
nat_bdr = None
src= fd.conditional(x**2 + y**2 < h_mesh**2,
                    0.1*n, 0.) # 10 m3/d/m3


# %%
# physical parameters
beta = 4.5e-8 # 1/Pa
rho = lambda x: 1000*(1 + beta*(x - 1e5))
mu = 1e-3
phi = 0.1
k = 1e-14*fd.Identity(dim)

g_ = -10 # None (without gravity)
g = fd.as_vector(tuple([0 for d in range(dim-1)]) + (0 if g_ is None else -g_,))


# %%
# set forms
# ---------

W = fd.FunctionSpace(mesh, "CG", 1)
q = fd.TestFunction(W)

p = fd.Function(W, name="p")
p0 = fd.Function(W, name="p0")

# The time step (just for initialization
dtc = fd.Constant(.0)

# bilinear forms
a = fd.inner(phi*rho(p), q)*fd.dx + \
    dtc*fd.inner(rho(p)*k/mu*fd.grad(p), fd.grad(q))*fd.dx

# gravity term
if g_ is not None:
    a -= dtc*fd.inner(fd.div(rho(p)*rho(p)/mu*k*g), q)*fd.dx

# linear forms
L = fd.inner(phi*rho(p0), q)*fd.dx

# ---------------
# add source term
if src is not None:
    L += dtc*fd.inner(src, q)*fd.dx

# --------------------------
# natural boundary condition
if nat_bdr is not None:
    L -= (dtc*fd.inner(nat_bdr, q) * fd.ds(LEFT) + 
          dtc*fd.inner(nat_bdr, q) * fd.ds(BOTTOM))

# -------------------------
# set up boundary condition
ess_bdr = [fd.DirichletBC(W, prd_well, RIGHT),
       fd.DirichletBC(W, prd_well, TOP)]


# nonlinear VP and solver
F = a - L
prob = fd.NonlinearVariationalProblem(F, p,
                                      bcs=ess_bdr)
solver = fd.NonlinearVariationalSolver(prob)


pfile = fd.File("plots/p.pvd")

# darcy velocity
V = fd.VectorFunctionSpace(mesh, "CG", 1, name="u")
u = fd.Function(V, name="u")
vfile = fd.File("plots/u.pvd")

p.interpolate(fd.Constant(1e5))

freq_res = 5
t = 0
it = 0
tsim = 1000
dt = tsim/50

dtc.assign(dt)
while t < tsim:
    it += 1
    t += dt
    print("* iteration= {}, time={}".format(it, t))
    solver.solve()
    
    p0.assign(p)
    if (it - 1) % freq_res == 0 or t == tsim:
        pfile.write(p, time=t)
        u.interpolate(-rho(p)*k/mu*fd.grad(p))
        vfile.write(u, time=t)
