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
from scipy import special
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
order = 2
refine = 2
nx = 10
ny = 4
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
v_inlet = [1e-2, .0]
cIn = 1.0 # injection conc
no_flow = 0.0
outflow = RIGHT

# problem parameters
tol = 1e-15
verbose = False
freq_res = 30
sim_time = 1e4 # simulation time
cfl = 0.25
tau_e = 100


# %%
# Process section (simulation)
# -----------------------------

# 2) Define mesh
print('* define mesh')
mesh = fd.RectangleMesh(int(nx * refine), int(ny * refine), Lx,
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

    # Define finite element spaces for the post - processing problem :
    #       l should be an integer satisfying 0 ≤ l ≤ k , where k is the
    #       approximation degree of the scalar solution .
    PP = fd.FunctionSpace(mesh,"DQ", order + 1)
    Pl = fd.FunctionSpace(mesh,"DQ", order)
else:
    DG1 = fd.FunctionSpace(mesh, "DG", order)
    vDG1 = fd.VectorFunctionSpace(mesh, "DG", order)
    Mh = fd.FunctionSpace(mesh, "HDiv Trace", order)

    DG0 = fd.FunctionSpace(mesh, "DG", 0)
    vDG = fd.VectorFunctionSpace(mesh, "DG", 1)

    # Define finite element spaces for the post - processing problem :
    #       l should be an integer satisfying 0 ≤ l ≤ k , where k is the
    #       approximation degree of the scalar solution .
    PP = fd.FunctionSpace(mesh,"DG", order + 1)
    Pl = fd.FunctionSpace(mesh,"DG", order)

W = DG1 * vDG1 * Mh
Wpp = PP * Pl

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
velocity.interpolate(fd.Constant(v_inlet))
vnorm = fd.sqrt(fd.dot(velocity, velocity))

# Diffusion tensor
Diff = fd.Identity(mesh.geometric_dimension())*Dm
# Diff = fd.Identity(mesh.geometric_dimension())*(Dm + d_t*vnorm) + \
#     fd.Constant(d_l-d_t)*fd.outer(velocity, velocity)/vnorm

# upwind ter
vn = 0.5*(fd.dot(velocity, n) + abs(fd.dot(velocity, n)))
cupw = fd.conditional(vn > 0, ch, lmbd_h)
# cupw = fd.conditional(fd.dot(velocity, n) > tol, ch, lmbd_h)

# stability
tau_d = fd.Constant(max([Dm, tau_e])) / h
tau_a = abs(fd.dot(velocity, n))
# tau_a = vn

# numerical flux
chat = lmbd_h
###########################################
qhat = qh + tau_d*(ch - chat)*n + velocity * chat + tau_a*(ch - chat)*n
# qhat = qh + tau*(ch - chat)*n + velocity * cupw
# qhat_n = fd.dot(qh, n) + tau*(ch - chat) + chat*vn

# ###########################################
# # Lax-Friedrichs/Roe form
# ch_a = 0.5*(ch('+') + ch('-'))
# qhat = qh + tau_d*(ch - chat)*n + velocity * ch_a + tau_a*(ch - ch_a)*n

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


# Scalar post-process

# Define elemental matrices / vectors for the local problems
pp , psi = fd.TrialFunctions (Wpp)
ww , phi = fd.TestFunctions (Wpp)

# linear system
a_pp = fd.inner ( fd.grad ( ww ) , fd.grad ( pp ) ) * fd.dx + fd.inner (ww , psi ) * fd.dx + \
    fd.inner ( phi , pp ) * fd.dx
L_pp = - fd.inner ( fd.grad ( ww ) , fd.inv(Diff) * qh ) * fd.dx + fd.inner ( phi , c0 ) * fd.dx


ch_pp = fd.Function(Wpp)


# %%
# 4) Solve problem
# solve

# set boundary conditions
bc = fd.DirichletBC(W.sub(2), cIn, inlet)
problem = fd.NonlinearVariationalProblem(a - L, w, bcs=bc)
rtol = 1e-4


# # scalar post-process
# ppproblem = fd.LinearVariationalProblem(a_pp, L_pp, ch_pp)


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


# # post-process
# ppsolver = fd.LinearVariationalSolver(ppproblem)


# -----------------
wave_speed = fd.conditional(fd.lt(np.abs(vnorm), tol), h/tol, h/vnorm)
# CFL
dt_ = fd.interpolate(wave_speed, DG1).dat.data.min()*cfl
outfile = fd.File(f"plots/adr-hdg-pp-n-{refine}-p-{order}.pvd", project_output=True)
dtc.assign(dt_)
dt = dt_

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

    # # scalar post-process
    # ppsolver.solve()

    # update sol.
    c0.assign(w.sub(0))

    # # acochambramento c (c<1) and (c>0)
    # val = c0.dat.data
    # if np.any((val > cIn) + (val < np.sqrt(tol))):
    #     val[val > cIn] = 1.0
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
        c_h.assign(c0)
        q_h.rename("q_h")

        # # scalar post-process
        # ch_star, lamb_pp = ch_pp.split()
        # ch_star.rename("ch_pp")

        A = fd.Tensor(a_pp)
        b = fd.Tensor(L_pp)
        ch_star = fd.assemble((A.inv * b).blocks[0])
        ch_star.rename("ch_star")

        # print results
        outfile.write(c_h, q_h, ch_star, time=t)
        print('-- write results @ t= {}'.format(t))

print(f'DOF = {W.dim()}')
print(f'DOF = {W.sub(2).dim()}')
print('* normal termination.')
