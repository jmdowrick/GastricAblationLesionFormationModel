import ufl
import numpy as np
import basix.ufl
from dolfinx import fem, mesh
from dolfinx.fem import petsc as PETSC
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI

class CellDeathSolver:
    def __init__(self, domain, T_func, params):
        self.mesh = domain.mesh
        self.T = T_func

        cell_type = basix.CellType(3)

        # Define space
        P1 = basix.ufl.element("CG", cell_type, 1)
        M = basix.ufl.mixed_element([P1, P1, P1])
        self.W = fem.functionspace(self.mesh, M)

        self.NUD = fem.Function(self.W) # current time step
        self.NUD_n = fem.Function(self.W) # previous time step

        self.initialise_cell_states()

        N, U, D = ufl.split(self.NUD)
        N_n, U_n, D_n = ufl.split(self.NUD_n)
        n, u, d = ufl.TestFunctions(self.W)

        # Thermal parameters
        DeltaE1 = float(params['celldeath']['DeltaE1'])
        DeltaE2 = float(params['celldeath']['DeltaE2'])
        DeltaE3 = float(params['celldeath']['DeltaE3'])
        A1 = float(params['celldeath']['A1'])
        A2 = float(params['celldeath']['A2'])
        A3 = float(params['celldeath']['A3'])
        R = float(params['celldeath']['R'])
        dt = params['simulation']['dt']

        k1 = A1*ufl.exp(-DeltaE1/(R*self.T))
        k2 = A2*ufl.exp(-DeltaE2/(R*self.T))
        k3 = A3*ufl.exp(-DeltaE3/(R*self.T))

        dNdt = (N - N_n)/dt
        dUdt = (U - U_n)/dt
        dDdt = (D - D_n)/dt

        F1 = (dNdt + k1*N + -k3*U) * n * ufl.dx
        F2 = (dUdt - k1*N + k3*U + k2*U) * u * ufl.dx
        F3 = (dDdt - k2*U) * d * ufl.dx

        self.F = F1 + F2 + F3

        petsc_options = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "none",
            "snes_atol": 1e-12,
            "snes_rtol": 1e-6,
            "ksp_error_if_not_converged": True,
            "ksp_type": "gmres",
            "ksp_rtol": 1e-8,
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "pc_hypre_boomeramg_max_iter": 1,
            "pc_hypre_boomeramg_cycle_type": "v",
        }

        self.problem = NonlinearProblem(
            F=self.F, 
            u=self.NUD,
            bcs=[],
            petsc_options_prefix="celldeath",
            petsc_options=petsc_options,
        )

    def solve_step(self):
        self.problem.solve()

    def advance_time(self):
        self.NUD_n.x.array[:] = self.NUD.x.array

    def initialise_cell_states(self):
        V_N, map_N = self.W.sub(0).collapse()
        self.NUD.x.array[map_N] = 1.0
        self.advance_time()
