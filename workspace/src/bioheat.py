import ufl
from dolfinx import fem
from src.electrostatics import calculate_sigma

class BioheatSolver:
    def __init__(self, domain, params, T_func, V_func):
        self.mesh = domain.mesh
        self.T_space = T_func.function_space

        # Current and previous temperature fields
        self.T = T_func
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
        T_ref = params['thermal']['T_ref']
        dt = params['simulation']['dt']

        # Variational form of Bioheat equation
        Q_p = 0.8*rho*cb*(T_ref - u)                  # Blood perfusion (heat loss)
        Q_m = 33800                                   # Metabolic heat generation
        E = -ufl.grad(V_func)
        Q_joule = calculate_sigma(self.T_n, params) \
          * ufl.inner(E, E)                           # Joule heating

        # Implicit Euler variational form
        F = (rho * cb * (u - self.T_n) / dt) * v * ufl.dx \
          + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
          - (Q_joule + Q_m + Q_p) * v * ufl.dx

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
