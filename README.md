# Firedrake Playground

This playground repo contains several scripts to solve some problems that I faced around or the ones I found interest, using [Firedrake](https://www.firedrakeproject.org/) framework, a very powerful FEM toolkit that provides a Python high-level syntax (more specifically, a Domain Specific Language - DSL) and automatic code generation.

# Available playground problems
* **Flow**
    * Mixed Finite element (Raviart-Thomas) for Darcy flow with spatially varying permeability ,inspired in MRST lognormal function.

* **Transport**
    * Transient linear advection-diffusion-reaction with a simple first-order reaction term. Available with the Crank-Nicolson time stepping and discretization example:
        - SUPG
        - SUPG + shock capturing stabilization
    * Two phases flow through porous media problem in a full implicit scheme.

* **Thermal**

