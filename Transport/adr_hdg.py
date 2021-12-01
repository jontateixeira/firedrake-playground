# -*- coding: utf-8 -*-
"""
Solves unsteady state advection-diffusion-reaction problem.

Strong form:

   dc/dt + div(c*u) - div(Diff*grad(c)) = f         in Omega

                       c = c_0  in Gamma_D

Mixed form:
        q + div(Diff*grad(c)) = 0         in Omega
    dc/dt + div(c*u) + div(q) = f         in Omega

Weak form:

Find c and q in W and W², such that:

 (inv(Diff)*qh, p) - (ch, div(p)) + <lmbda, [p]> = 0

(r,(c - c0))/dt - (grad(r), u*ch)
  + (grad(r), [qh]) + <r, u*lmbda*n> + <r, qĥ.n> = (r,f)

                                    <mu,lmbda>_D = <mu, c_0>_D

                             <mu,qĥ + lmbda*u>_D = <mu, g_0>_N

where:
    qĥ = qh + tau*(ch - ĉh).n
    ĉh = lmbda

for all r, p and lmbda in W', W'² and Mh.

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
# import plot_sparsity as sparse_plt
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
refine = 3
nx = 20
ny = 5
Lx = 100 # m
Ly = 20 # m
quad_mesh = False


# materials parameters
# reaction/transport parameter
Dm = 1e-9 # diffusion (m²/s)
d_l = 2*Lx/ny # longitidinal dispersion (m)
d_t = 0.5*d_l # transversal dispersion (m)
K = 0.00 # reaction rate (mol/m³/s)


# boundary conditions
inlet = LEFT
v_inlet = fd.Constant([1e-3, .0])
cIn = 1.0 # injection conc
no_flow = 0.0
outflow = RIGHT

# problem parameters
tol = 1e-15
verbose = False
freq_res = 100
sim_time = 1e5 # simulation time
dtime = 3e3
cfl = 0.5


# %%
# Process section (simulation)
# -----------------------------

# 2) Define mesh
print('* define mesh')
mesh = fd.RectangleMesh(nx * refine, ny * refine, Lx,
                        Ly, quadrilateral=quad_mesh)

if verbose:
    fd.triplot(mesh)
    plt.legend()
    plt.savefig("plots/adr_dg_mesh.png")

# 3) Setting problem (FunctionSpace, Init.Bound.Condition, VariationalForms)

# 3.1) Set Function spaces
if quad_mesh:
    DG1 = fd.FunctionSpace(mesh, "DQ", order)
    vDG1 = fd.VectorFunctionSpace(mesh, "DQ", order)
    Mh = fd.FunctionSpace(mesh, "HDiv Trace", order)

    DG0 = fd.FunctionSpace(mesh, "DQ", 0)
    vDG = fd.VectorFunctionSpace(mesh, "DQ", 1)
else:
    DG1 = fd.FunctionSpace(mesh, "DG", order)
    vDG1 = fd.VectorFunctionSpace(mesh, "DG", order)
    Mh = fd.FunctionSpace(mesh, "HDiv Trace", order)

    DG0 = fd.FunctionSpace(mesh, "DG", 0)
    vDG = fd.VectorFunctionSpace(mesh, "DG", 1)

W = DG1 * vDG1 * Mh

# 3.1) Define trial and test functions
w = fd.Function(W)
w.assign(0.0)
ch, qh, lmbd_h = fd.split(w)
wh, vh, mu_h = fd.TestFunctions(W)


# 3.2) Set initial conditions
# ---- previous solution
# concentrations
c0 = fd.Function(DG1, name="c0")
c0.assign(0.0)

# ----------------------
# 3.4) Variational Form
# ----------------------
# coefficients
dt = np.sqrt(tol)
dtc = fd.Constant(dt)
n = fd.FacetNormal(mesh)
h = fd.sqrt(2) * fd.CellVolume(mesh) / fd.CellDiameter(mesh)


# advective velocity
velocity = fd.Function(vDG1)
velocity.interpolate(v_inlet)
vnorm = fd.sqrt(fd.dot(velocity, velocity))

# upwind term
vn = 0.5*(fd.dot(velocity, n) + abs(fd.dot(velocity, n)))
cupw = fd.conditional(fd.dot(velocity, n) > 0, ch, lmbd_h)

# Diffusion tensor
Diff = fd.Identity(mesh.geometric_dimension())*Dm
# Diff = fd.Identity(mesh.geometric_dimension())*(Dm + d_t*vnorm) + \
#     fd.Constant(d_l-d_t)*fd.outer(velocity, velocity)/vnorm

# stability
# tau = fd.Constant(1.0) / h + abs(fd.dot(velocity, n))
tau = fd.Constant(5) / h + vn

# numerical flux
chat = lmbd_h
qhat = qh + tau*(ch - chat)*n + velocity * chat
# qhat_n = fd.dot(qh, n) + tau*(ch - chat) + chat*vn


a_u = (fd.inner(fd.inv(Diff)*qh, vh)*fd.dx -
       ch*fd.div(vh)*fd.dx +
       # internal faces
       fd.jump(lmbd_h*vh, n)*fd.dS +
       # other faces
       lmbd_h*fd.inner(vh, n)*fd.ds(outflow) +
       lmbd_h*fd.inner(vh, n)*fd.ds(TOP) +
       lmbd_h*fd.inner(vh, n)*fd.ds(BOTTOM)
       )

# Dirichlet faces
L_u = - fd.Constant(cIn)*fd.inner(vh, n)*fd.ds(inlet)


a_c = (wh*(ch - c0)/dtc*fd.dx -
       fd.inner(fd.grad(wh), qh + ch * velocity)*fd.dx +
       wh("+")*fd.jump(qhat, n)*fd.dS +
       wh * fd.inner(qhat, n) * fd.ds)

L_c = 0

# transmission boundary condition
F_q = mu_h("+")*fd.jump(qhat, n)*fd.dS + \
    mu_h*fd.inner(qhat, n)*fd.ds(outflow) + \
    mu_h*fd.inner(qhat, n)*fd.ds(TOP) + \
    mu_h*fd.inner(qhat, n)*fd.ds(BOTTOM)

G_q = mu_h*fd.inner(Diff*fd.grad(ch) + velocity * ch, n)*fd.ds(outflow) + \
    mu_h*fd.Constant(no_flow)*fd.ds(TOP) + \
    mu_h*fd.Constant(no_flow)*fd.ds(BOTTOM)

# G_q = 0

a = a_u + a_c + F_q
L = L_u + L_c + G_q


# %%
# 4) Solve problem
# solve

# set boundary conditions
bc = fd.DirichletBC(W.sub(2), cIn, inlet)
problem = fd.NonlinearVariationalProblem(a - L, w, bcs=bc)
rtol = 1e-4


# sparse_plt.plot_matrix(a, bcs=bc)
# plt.show()
# sparse_plt.plot_matrix_hybrid_multiplier_spp(a, bcs=bc)
# plt.show()

hybrid_solver_params = {
    "snes_type": "ksponly",
    "mat_type": "matfree",
    "pmat_type": "matfree",
    "ksp_type": "preonly",
    "pc_type": "python",
    # Use the static condensation PC for hybridized problems
    # and use a direct solve on the reduced system for lambda_h
    "pc_python_type": "firedrake.SCPC",
    "pc_sc_eliminate_fields": "0, 1",
    "condensed_field": {"ksp_type": "preonly",
                        "pc_type": "lu",
                        "pc_factor_mat_solver_type": "mumps"}
}

solver = fd.NonlinearVariationalSolver(problem,
                                       solver_parameters=hybrid_solver_params)

# -----------------
wave_speed = fd.conditional(fd.lt(np.abs(vnorm), tol), h/tol, h/vnorm)
# CFL
dt_ = fd.interpolate(wave_speed, DG1).dat.data.min()*cfl
outfile = fd.File(f"plots/adr-hdg-n-{refine}-p-{order}.pvd", project_output=True)
dt = dt_ # dtime
dtc.assign(dt_)

# %%
# solve problem

# initialize timestep
t = 0.0
it = 0

p = 0
while t < sim_time:
    # check dt
    dt = np.min([sim_time - t, dt])

    # move next time step
    it += 1
    t += dt
    print("* iteration= {:4d}, dtime= {:8.6f}, time={:8.6f}".format(it, dt, t))

    # tracer
    dtc.assign(dt)
    solver.solve()

    # update sol.
    c0.assign(w.sub(0))

    # # acochambramento c (c<1) and (c>0)
    # val = c0.dat.data
    # if np.any((val > cIn) + (val < np.sqrt(tol))):
    #     val[val > cIn] = 1.
    #     val[val < np.sqrt(tol)] = tol
    #     c0.dat.data[:] = val

    # %%
    # 5) print results
    if it % freq_res == 0 or t == sim_time:
        if verbose:
            contours = fd.tripcolor(c0)
            plt.xlabel(r'$x$')
            plt.ylabel(r'$y$')
            plt.colorbar(contours)
            plt.show()

        # Computed flux, scalar, and trace
        c_h, q_h, lamb = w.split()
        c_h.rename("c_h")
        q_h.rename("q_h")

        # print results
        outfile.write(c_h, q_h, time=t)
        print('-- write results @ t= {}'.format(t))

print(f'DOF = {W.dim()}')
print(f'DOF = {W.sub(2).dim()}')
print('* normal termination.')
