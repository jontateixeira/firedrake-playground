#!/home/john/opt/firedrake/bin/python3
# -*- coding: utf-8 -*-
"""
Mesh-Playground: Creating simples 1D and 2D Meshes
"""

from firedrake import (IntervalMesh, UnitIntervalMesh, PeriodicIntervalMesh,
                       RectangleMesh, SquareMesh,
                       UnitSquareMesh, PeriodicRectangleMesh, triplot,
                       PeriodicSquareMesh, PeriodicUnitSquareMesh)
import matplotlib.pyplot as plt
import numpy as np

# creating a square unit domain
quad = True    # quadrilateral elements
nx = ny = 10
L = Lx = Ly = 2.0

# see utility_meshes.py
#  The boundary surfaces are numbered as follows: (except for Periodic Mesh)
#  * 1: plane x == 0
#  * 2: plane x == Lx
#  * 3: plane y == 0
#  * 4: plane y == Ly
#  * 5: plane z == 0
#  * 6: plane z == Lz


# OneDimensional Meshes

# -> IntervalMesh
# ---------------
# 1D mesh support ncells, Length (left coords), right coords (opt.)
#  m= IntervalMesh(nx,-L,L)
m = IntervalMesh(nx, L)
xIM = m.coordinates.dat.data


# -> UnitIntervalMesh
# -------------------
m = UnitIntervalMesh(nx)
xUIM = m.coordinates.dat.data


# -> PeriodicInternalMesh
# -----------------------
m = PeriodicIntervalMesh(nx, L)
xPIM = m.coordinates.dat.data

y = np.ones([nx+1, ])
plt.plot(xIM, 0*y, color="black", marker="o", label="IntervalMesh")
plt.plot(xUIM, 1*y, color="red", marker="o", label="UnitIntervalMesh")
plt.plot(xPIM, 2*np.ones(np.shape(xPIM)), color="blue",
         marker="o", label="PeriodicInternalMesh")
plt.legend()
plt.savefig("plots/oneDimensionMesh.png")


# TwoDimensional Meshes
# -> RectangleMesh
# ----------------
# support quadrilateral input (True or False)
mesh = RectangleMesh(nx, ny, Lx, Ly)
triplot(mesh)
plt.legend()
plt.savefig("plots/RectangleMesh.png")


# -> SquareMesh
# -------------
# using SquareMesh (possible use of quadrilateral elements)
mesh = SquareMesh(nx, ny, L, quadrilateral=quad)
triplot(mesh)
plt.legend()
plt.savefig("plots/SquareMesh.png")


# -> UnitSquareMesh
# -----------------
# possible use of quadrilateral elements
mesh = UnitSquareMesh(nx, ny, quadrilateral=not quad)
triplot(mesh)
plt.legend()
plt.savefig("plots/UnitSquareMesh.png")


# -> PeriodicRectangleMesh
# ------------------------
# possible use of quadrilateral elements
mesh = PeriodicRectangleMesh(nx+1, ny, Lx, Ly)
triplot(mesh)
plt.legend()
plt.savefig("plots/PeriodicRectangleMesh.png")


# -> PeriodicSquareMesh
# ------------------------
# possible use of quadrilateral elements
mesh = PeriodicSquareMesh(nx, ny, Lx, direction="x")
triplot(mesh)
plt.legend()
plt.savefig("plots/PeriodicSquareMesh.png")


# -> PeriodicUnitSquareMesh
# --------------------------
# possible use of quadrilateral elements
mesh = PeriodicUnitSquareMesh(nx, ny, direction="y")
triplot(mesh)
plt.legend()
plt.savefig("plots/PeriodicUnitSquareMesh.png")
