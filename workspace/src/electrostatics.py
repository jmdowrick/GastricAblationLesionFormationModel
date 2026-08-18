import ufl
import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc as PETSC
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

def calculate_sigma(T, params):
    return params['electrical']['sigma_0'] * (1 + params['thermal']['eta_sigma']*(T - params['thermal']['T_ref']))

class ElectrostaticSolver:
    def __init__(self, domain, V_func, T_func, Ftot_func, params, initial_current=50.0):
        self.mesh = domain.mesh
        self.V_space = V_func.function_space
        self.V = V_func

        dim = domain.mesh.topology.dim

        x = ufl.SpatialCoordinate(self.mesh)
        dx = ufl.Measure("dx", domain=self.mesh)
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=domain.facet_tags) 
        catheter_tags = domain.physical_groups['catheter'].tag
        
        # 1. Boundary conditions
        # A. Dirichlet BC
        ground_facets = domain.facet_tags.find(domain.physical_groups['base'].tag)
        ground_dofs = fem.locate_dofs_topological(self.V_space, dim-1, ground_facets)
        bc_ground = fem.dirichletbc(fem.Constant(self.mesh, 0.0), ground_dofs, self.V_space)
        self.bcs = [bc_ground]

        # B. Neumann BC
        self.E_applied = fem.Constant(self.mesh, float(initial_current))
        
        # 2. Setup standard variational form 
        u = ufl.TrialFunction(self.V_space)
        v = ufl.TestFunction(self.V_space)
        self.sigma = calculate_sigma(T_func, params)

        Ftot_inplane = ufl.as_matrix([[Ftot_func[0,0], Ftot_func[0,1]],
                                       [Ftot_func[1,0], Ftot_func[1,1]]])
        F_1 = ufl.inv(Ftot_inplane)
        F_1T = ufl.transpose(F_1)
        sigma_tensor = self.sigma * ufl.Identity(dim)

        self.a = ufl.inner(F_1 * self.sigma * F_1T * ufl.grad(u), ufl.grad(v)) * x[0] * dx
        self.L = self.E_applied * v * x[0] * ds(catheter_tags)
        
        self.problem = LinearProblem(
            self.a, self.L, bcs=self.bcs, u=self.V,
            petsc_options_prefix="electrostatic",
            petsc_options={
                "ksp_type": "cg",
                "pc_type": "hypre",
                "ksp_error_if_not_converged": True,
                }
          )
        
        # 3. Pre-compile the UFL form for total dissipated power
        self.E = -ufl.grad(self.V)
        power_integrand = 2 * np.pi * self.sigma * ufl.inner(ufl.grad(self.V), ufl.grad(self.V)) * x[0] * dx
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
                print(f"  Adjusting Current: {self.E_applied.value:.2f} V -> {self.E_applied.value / lam:.2f} A")
                
            # Update the applied voltage via the Constant
            self.E_applied.value = self.E_applied.value / lam
            
            # Re-solve the system with the new boundary condition
            self.solve()

    def solve(self):
        self.problem.solve()
