import ufl
import numpy as np
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

def calculate_sigma(T, params):
    return params['electrical']['sigma_0'] * (1 + params['thermal']['eta_sigma']*(T - params['thermal']['T_ref']))

class ElectrostaticSolver:
    def __init__(self, mesh, V_func, T_func, params, active_electrode_facets, ground_facets, initial_voltage=50.0):
        self.mesh = mesh
        self.V_space = V_func.function_space
        self.V = V_func
        
        # 1. Create Dirichlet BCs 
        self.V_applied = fem.Constant(mesh, float(initial_voltage))
        self.V_ground = fem.Constant(mesh, 0.0)
        
        active_dofs = fem.locate_dofs_topological(self.V_space, 2, active_electrode_facets)
        ground_dofs = fem.locate_dofs_topological(self.V_space, 2, ground_facets)
        
        bc_active = fem.dirichletbc(self.V_applied, active_dofs, self.V_space)
        bc_ground = fem.dirichletbc(self.V_ground, ground_dofs, self.V_space)
        self.bcs = [bc_active, bc_ground]
        
        # 2. Setup standard variational form 
        u = ufl.TrialFunction(self.V_space)
        v = ufl.TestFunction(self.V_space)
        self.sigma = calculate_sigma(T_func, params)
        
        self.a = self.sigma * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        self.L = fem.Constant(mesh, 0.0) * v * ufl.dx
        
        self.problem = LinearProblem(
            self.a, self.L, bcs=self.bcs, u=self.V,
            petsc_options_prefix="electrostatic",
            petsc_options={
                "ksp_type": "cg",
                "pc_type": "hypre"
        })
        
        # 3. Pre-compile the UFL form for total dissipated power
        # Power = integral( sigma * E^2 ) dx
        self.E = -ufl.grad(self.V)
        power_integrand = self.sigma * ufl.inner(self.E, self.E) * ufl.dx
        self.power_form = fem.form(power_integrand)

    def enforce_power_constraint(self, target_power, tol=0.01):
        # Assemble local power and sum across all MPI processes
        local_power = fem.assemble_scalar(self.power_form)
        current_power = self.mesh.comm.allreduce(local_power, op=MPI.SUM)
        
        # Prevent division by zero if initialization hasn't happened
        if current_power < 1e-12:
            return
            
        # Compute ratio (lambda)
        lam = np.sqrt(current_power / target_power)
        
        if abs(lam - 1.0) > tol:
            if self.mesh.comm.rank == 0:
                print(f"  Power constraint update: Ratio {abs(lam-1.0):.4f}")
                print(f"  Target/Current Power: {target_power:.2f} / {current_power:.2f} W")
                print(f"  Adjusting Voltage: {self.V_applied.value:.2f} V -> {self.V_applied.value / lam:.2f} V")
                
            # Update the applied voltage via the Constant
            self.V_applied.value = self.V_applied.value / lam
            
            # Re-solve the system with the new boundary condition
            self.solve()

    def solve(self):
        self.problem.solve()
