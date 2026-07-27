import ufl
from dolfinx import fem
from src.electrostatics import calculate_sigma

class BioheatSolver:
    def __init__(self, mesh, params, V_func, dt):
        self.mesh = mesh

        self.T_space = fem.functionspace(mesh, ("Lagrange", 1))

        # Current and previous temperature fields
        self.T = fem.Function(self.T_space)
        self.T_n = fem.Function(self.T_space)

        # Initialise both to body temperature
        self.T.x.array[:] = params['thermal']['T_ref']
        self.T_n.x.array[:] = params['thermal']['T_ref']

        u = ufl.TrialFunction(self.T_space)
        v = ufl.TestFunction(self.T_space)

        # Tissue properties
        rho = params['thermal']['rho_b']
        cb = params['thermal']['cb']
        kappa = params['thermal']['kappa']

        # Define Joule heating purely in UFL using the voltage function
        # sigma must match the definition in electrostatics
        sigma = calculate_sigma(self.T_n, params)

        Q_joule = sigma * ufl.inner(ufl.grad(V_func), ufl.grad(V_func))

        # Implicit Euler variational form
        F = (rho * cb * (u - self.T_n) / dt) * v * ufl.dx \
          + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
          - Q_joule * v * ufl.dx

        self.a, self.L = ufl.system(F)
        self.bcs = []

        self.problem = fem.petsc.LinearProblem(
            self.a, self.L, bcs=self.bcs, u=self.T,
            petsc_options_prefix="bioheat",
            petsc_options={
                "ksp_type": "cg",
                "pc_type": "jacobi"
        })

    def solve_step(self):
        self.problem.solve()
        
    def advance_time(self):
        self.T_n.x.array[:] = self.T.x.array
