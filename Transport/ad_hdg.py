# -*- coding: utf-8 -*-

"""
Solves linear steady state advection-diffusion problem (example 2 adapted of
the nguyen et.al 2009).

Strong form:

   div(c*u) - div(Diff*grad(u)) = f         in Omega

                              u = u_0  in Gamma_D
         (c*u - Diff*grad(u)).n = gN   in Gamma_N

Mixed form:
        q + div(Diff*grad(u)) = 0         in Omega
            div(c*u) + div(q) = f         in Omega

Weak form:

Find uh, qh, lmbda in V, W² and Mh(0), such that:

 (inv(Diff)*qh, w) - (ch, div(w)) + <lmbda, [w]> = 0

 - (grad(v), c*uh) + (grad(v), [qh]) + 
                      <v, u*lmbda*n> + <v, qĥ.n> = (v,f)

                                    <mu,lmbda>_D = <mu, c_0>_D

                             <mu,qĥ + lmbda*c>_D = <mu, g_0>_N

where:
    qĥ = qh + tau*(uh - ûh).n
    ûh = lmbda

for all v, w and mu in W', W'² and Mh.

Model problem:

 ------4------
 |           |
 1           2
 |           |
 ------3------

velocity profile:
    c(x,y) = (25, 25)

u exact
                (1 - exp(cx*(x-1)))*(1 - exp(cy*(y-1))) 
    u = x * y * ---------------------------------------
                    ((1 - exp(-cx))*(1 - exp(-cy)))

boundary conditions:
c(x, t) = 0       on Gamma_{1, 2, 3, 4}
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
order = 3
refine = 1
nx = 4
ny = 4
Lx = 1.0 # m
Ly = 1.0 # m
from_file = True
quad_mesh = False
post_process = True
tau_e = 1

# boundary conditions
c0 = 0.0  # boundary concentration

# problem parameters
verbose = False
Dm = 1.0  # diffusion (m²/s)
c = (25,25)


# %%
# Process section (simulation)
# -----------------------------

# 2) Define mesh
print('* define mesh')
if from_file:
    mesh = fd.Mesh("nguyen.msh")
    mesh.init()
else:
    mesh = fd.RectangleMesh(nx * refine, ny * refine, Lx, Ly, quadrilateral=quad_mesh, diagonal="right")

x, y = fd.SpatialCoordinate(mesh)
if verbose:
    fd.triplot(mesh)
    plt.legend()
    plt.savefig("plots/cd_hdg_mesh.png")

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

# ----------------------
# 3.4) Variational Form
# ----------------------
# coefficients
n = fd.FacetNormal(mesh)
h = fd.sqrt(2) * fd.CellVolume(mesh) / fd.CellDiameter(mesh)

# advective velocity
velocity = fd.Function(vDG1)
velocity.interpolate(fd.Constant(c))
vnorm = fd.sqrt(fd.dot(velocity, velocity))

# source
u_exact = x * y * (1 - fd.exp(c[0]*(x-1)))*(1 - fd.exp(c[1]*(y-1))) / ((1 - fd.exp(-c[0]))*(1 - fd.exp(-c[1])))
source = fd.div(velocity * u_exact) - fd.div(Dm*fd.grad(u_exact))

# stability
tau_d = fd.Constant(max([Dm, tau_e])) / h
tau_a = abs(fd.dot(velocity, n))
# tau = fd.Constant(1.0) / h + vn

# numerical flux
qhat = qh + tau_d*(ch - lmbd_h)*n
fhat = velocity * lmbd_h + tau_a*(ch - lmbd_h)*n

a_u = (fd.inner(fd.inv(Dm)*qh, vh)*fd.dx -
       ch*fd.div(vh)*fd.dx +
       # internal faces
       fd.jump(lmbd_h*vh, n)*fd.dS +
       # other faces
       lmbd_h*fd.inner(vh, n)*fd.ds
       )

# Dirichlet faces
L_u = 0


a_c = (- fd.inner(fd.grad(wh), qh + ch * velocity)*fd.dx + 
       wh('+')*fd.jump(qhat + fhat, n)*fd.dS +
       wh*fd.inner(qhat + fhat, n)*fd.ds -
       wh*source*fd.dx)

L_c = 0

# transmission boundary condition
F_q = (
    mu_h('+')*fd.jump(qhat + fhat, n)*fd.dS + \
    mu_h*fd.Constant(0.0)*fd.ds #+ \
    # next lines only added if impose {in/out/no}flow bdr cond. (Neumann type)
    #mu_h*fd.inner(qhat, n)*fd.ds({in/out/no}flow)
    )

# G_q = mu_h*fd.inner(Dm*fd.grad(ch) + velocity * ch, n)*fd.ds({in/out/no}flow) # 
G_q = mu_h*lmbd_h*fd.ds


a = a_u + a_c + F_q
L = L_u + L_c + G_q


if post_process:
    # Scalar post-process
    # Define elemental matrices / vectors for the local problems
    pp , psi = fd.TrialFunctions (Wpp)
    ww , phi = fd.TestFunctions (Wpp)

    # linear system
    a_pp = fd.inner ( fd.grad ( ww ) , Dm*fd.grad ( pp ) ) * fd.dx + fd.inner (ww , psi ) * fd.dx + \
        fd.inner ( phi , pp ) * fd.dx
    L_pp = - fd.inner ( fd.grad ( ww ) , qh ) * fd.dx + fd.inner ( phi , ch ) * fd.dx


    ch_pp = fd.Function(Wpp)
    uh_pp = fd.Function(PP, name="c-star")

    # scalar post-process
    ppproblem = fd.LinearVariationalProblem(a_pp, L_pp, ch_pp)
    ppsolver = fd.LinearVariationalSolver(ppproblem)
    A = fd.Tensor(a_pp)
    b = fd.Tensor(L_pp)


# %%
# 4) Solve problem
# solve

# set boundary conditions
bc = fd.DirichletBC(W.sub(2), c0, "on_boundary") # this really need?
problem = fd.NonlinearVariationalProblem(a - L, w, bcs=bc)
rtol = 1e-4

hybrid_solver_params = {
    "snes_type": "ksponly",
    "mat_type": "matfree",
    "pmat_type": "matfree",
    "ksp_type": "preonly",
    "pc_type": "python",
    # Use the static condensation PC for hybridized problems
    # and use a direct solve on the reduced system for lambda_h
    "pc_python_type": "firedrake.SCPC",
    "pc_sc_eliminate_fields": "1, 0",
    "condensed_field": {"ksp_type": "preonly",
                        "pc_type": "lu",
                        "pc_factor_mat_solver_type": "mumps"},
}

solver = fd.NonlinearVariationalSolver(problem,
                                       solver_parameters=hybrid_solver_params)



solver.solve()

if post_process:
    # scalar post-process
    ppsolver.solve()
    uh_pp = fd.assemble((A.inv * b).blocks[0])
    uh_pp.rename("ch_star")


# Computed flux, scalar, and trace
c_h, q_h, lamb = w.split()
c_h.rename("c_h")
q_h.rename("q_h")

ue = fd.Function(DG1, name="exact").interpolate(u_exact)
if post_process:
    # scalar post-process
    ch_star, lamb_pp = ch_pp.split()
    ch_star.rename("ch_pp")

    # print results
    fd.File(f"plots/ad-hdg-nguyen-pp-n-{refine}-p-{order}.pvd").write(c_h, q_h, ch_star, uh_pp, ue)
else:
    # print results
    fd.File(f"plots/ad-hdg-nguyen-n-{refine}-p-{order}.pvd").write(c_h, q_h, ue)

print('* normal termination.')
