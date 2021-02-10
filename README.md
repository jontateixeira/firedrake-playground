# Firedrake Playground

This playground repo contains several scripts to solve some problems that I faced around or the ones I found interest, using [Firedrake](https://www.firedrakeproject.org/) framework, a very powerful FEM toolkit that provides a Python high-level syntax (more specifically, a Domain Specific Language - DSL) and automatic code generation.

# Available playground problems
* **Flow**
    * Galerkin Darcy continuity (compressible hydrostatic) equation.
    * Mixed finite element (raviart-thomas) for darcy flow with spatially varying permeability ,inspired in mrst lognormal function.
    * Mixed finite element (raviart-thomas) for linear stokes-brinkman problem. 


* **Transport**
    * Transient linear advection-diffusion-reaction with a simple first-order reaction term. Available with the Crank-Nicolson time stepping and discretization example:
        - SUPG
        - SUPG + shock capturing stabilization
        - DG
    * Two phases flow through porous media problem in a full implicit scheme.

* **Thermal**

