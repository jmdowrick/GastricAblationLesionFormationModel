import ufl
import numpy
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import NonlinearProblem
from src.parameters import load_parameters

params = load_parameters('parameters.yml')
