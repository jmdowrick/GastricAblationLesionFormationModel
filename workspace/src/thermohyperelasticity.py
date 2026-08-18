import numpy as np
import ufl
import basix.ufl
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem


def convert_grams_to_newtons(grams: float) -> float:
  return 0.00980665 * grams


def convert_diameter_to_area(diameter: float) -> float:
    return np.pi * (diameter / 2.0) ** 2


class ThermoHyperElasticitySolver:
    def __init__(self, domain, T_func, CD_func, Ftot_func, params):
        self.mesh = domain.mesh
        self.dim = domain.mesh.topology.dim
        cell_type = basix.CellType(self.dim)

        x = ufl.SpatialCoordinate(self.mesh)
        dx = ufl.Measure("dx", domain=self.mesh)
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=domain.facet_tags)
        catheter_tag = domain.physical_groups['catheter'].tag

        # -----
        # Function spaces (Taylor-Hood P2-P1 mixed element)
        # -----
        P2 = basix.ufl.element("Lagrange", cell_type, 2, shape=(self.dim,))  # Lagrange 2nd order element for displacement
        P1 = basix.ufl.element("Lagrange", cell_type, 1)              # Lagrange 1st order element for pressure
        M = basix.ufl.mixed_element([P2, P1])                         # Mixed pressure-displacement elements
        self.W = fem.functionspace(self.mesh, M)

        self.up = fem.Function(self.W)     
        vq = ufl.TestFunction(self.W)
        dup = ufl.TrialFunction(self.W) # direction for Jacobian?
        v, q = ufl.split(vq)
        u, p = ufl.split(self.up)

        self.Ftot = Ftot_func # shared deformation-gradient output

        # -----
        # Coupled fields
        # -----

        # Temperature
        T = T_func
        dT = T - float(params['thermal']['T_ref'])

        # Cell death
        N, U, D = ufl.split(CD_func)
        LuLn = float(params['celldeath']['LuLn'])
        LdLn = float(params['celldeath']['LdLn'])
        shrink = 1 - N - LuLn*U - LdLn*D

        # -------------------
        # Material parameters
        # -------------------

        # Mechanical
        K = float(params['mechanical']['K'])
        c1 = float(params['mechanical']['c1'])
        c2 = float(params['mechanical']['c2'])

        # Thermo-mechanical
        alfa = float(params['thermal']['alfaF'])
        beta = float(params['thermal']['beta'])

        # ---------------------------
        # Hyperelasticity formulation
        # ---------------------------

        # Kinematics
        I = ufl.Identity(3)

        Phi = 1 + alfa*dT

        grad_u = ufl.grad(u)
        u_r = u[0]

        #u_r_over_r = ufl.conditional(ufl.gt(x[0], 1e-12), u_r / x[0], grad_u[0, 0])

        F_tot = ufl.as_matrix([
            [1 + grad_u[0, 0], grad_u[0, 1], 0],
            [grad_u[1, 0], 1 + grad_u[1, 1], 0],
            [0, 0, 1 + u_r / (x[0]+1e-9)]
        ])
        J = ufl.det(F_tot)

        J_theta = Phi**3
        J_mech = J/J_theta

        F_theta = Phi*I                                 # Thermal Deformation gradient
        F_mech = F_tot*ufl.inv(F_theta)                 # Mechanical Deformation gradient
        F_mech_1 = ufl.inv(F_mech)                      # Inverse Mechanical Deformation gradient
        F_mech_1T = ufl.transpose(F_mech_1)             # Inverse Mechanical Transpose Deformation gradient
        F_mechbar = ( J_mech**(-1/3) )*F_mech           # Modified Mechanical Deformation gradient
        F_1 = ufl.inv(F_tot)                            # Inverse Deformation gradient
        F_1T = ufl.transpose(F_1)                       # Inverse Transpose Deformation gradient
        C = ufl.transpose(F_tot)*F_tot                  # Right Cauchy-Green tensor
        C_mech = ufl.transpose(F_mech)*F_mech           # Mechanical Right Cauchy-Green tensor
        C_mechbar = ufl.transpose(F_mechbar)*F_mechbar  # Modified Mechanical Right Cauchy-Green tensor

        # Modified invariants of deformation tensors
        I1bar = ufl.tr(C_mechbar)
        I2bar = 0.5*(I1bar**2 - ufl.tr(C_mechbar*C_mechbar))

        # Strain tensors (for post processing)
        E = 0.5*(C - I)
        e = F_1T*E*F_1

        # Strain energy density
        psiVol = (K/4)*(J_mech**2-1-2*ufl.ln(J_mech))+ p*(J/J_theta - 1)-(p**2)/(2*K)   # Incompressibility constraint using Perturbed Lagrangian method
        psiIso = c1*(I1bar-3) + c2*(I2bar-3)                          # Deviatoric psi for the matrix (Mooney-Rivlin)
        psiThermal = -3*(alfa-beta*shrink)*K*dT*(ufl.ln(J)/J)       # Thermal strain energy including both expansion (alfa) and shrinkage (-beta*shrink)             

        psi = psiVol + psiIso +psiThermal

        # Stress tensors (for post processing)
        Svol = J_mech*(K/2)*(J_mech - 1/J_mech)*ufl.inv(C_mech)
        Siso = 2*(J_mech**(-2/3))*((-1/3)*(c1*I1bar + 2*c2*I2bar)*ufl.inv(C_mech) + (c1+c2*I1bar)*I - c2*C_mechbar)
        Sthermal = F_mech_1*(-(3/2)*(alfa-beta*shrink)*K*dT*(1/J)*(1-ufl.ln(J)))*F_mech_1T

        S_mech = Svol + Siso + Sthermal    # Mechanical 2nd Piola-Kirchoff stress tensor  
        S = Phi**(-2)*S_mech               # 2nd Piola-Kirchoff stress tensor
        PK = F_tot*S                       # 1st Piola-Kirchoff stress tensor
        SC = (1/J)*PK*ufl.transpose(F_tot) # Cauchy stress tensor

        # -------
        # Pressure displacement mixed formulation for hyperelasticity
        # -------

        self.B_hyp = fem.Constant(self.mesh, np.zeros(self.dim))     # Body force per unit volume
        self.T_hyp = fem.Constant(self.mesh, np.zeros(self.dim))     # Traction force on the boundary

        PiExt = ufl.dot(self.B_hyp, u) * x[0] * dx + ufl.dot(self.T_hyp, u) * x[0] * ds(catheter_tag)
        PiInt = psi * x[0] * dx
        Pi = PiInt - PiExt

        # First variation of Pi (directional derivative about u in the direction of v)
        F = ufl.derivative(Pi, self.up, vq)

        # Compute Jacobian of F
        J_form = ufl.derivative(F, self.up, dup)

        # -----
        # Boundary conditions
        # -----

        base_facets = domain.facet_tags.find(domain.physical_groups['base'].tag)
        V_disp, _ = self.W.sub(0).collapse()
        clamp_dofs = fem.locate_dofs_topological((self.W.sub(0), V_disp), self.dim-1, base_facets)
        zero_disp = fem.Function(V_disp)
        self.bcs = [fem.dirichletbc(zero_disp, clamp_dofs, self.W.sub(0))]

        # axisymmetric BC - prevent radial displacement at the axis.
        if self.dim == 2:
            axis_facets = domain.facet_tags.find(domain.physical_groups['axis'].tag)
            V_r, _ = self.W.sub(0).sub(0).collapse()
            axis_dofs = fem.locate_dofs_topological((self.W.sub(0).sub(0), V_r), self.dim - 1, axis_facets)
            zero_ur = fem.Function(V_r)
            self.bcs.append(fem.dirichletbc(zero_ur, axis_dofs, self.W.sub(0).sub(0)))

        # -----
        # Define problem 
        # -----

        petsc_options = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "none",
            "snes_monitor": None,
            "snes_atol": 1e-8,
            "snes_rtol": 1e-8,
            "snes_stol": 1e-8,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

        self.problem = NonlinearProblem(
            F=F, u=self.up, bcs=self.bcs, J=J_form,
            petsc_options_prefix="thermohyperelasticity",
            petsc_options=petsc_options,
        )

        self.Ftot_expr = fem.Expression(
            F_tot, 
            self.Ftot.function_space.element.interpolation_points          
        )

        self._update_Ftot()

    def solve_step(self):
        self.problem.solve()
        self._update_Ftot()

    def _update_Ftot(self):
        self.Ftot.interpolate(self.Ftot_expr)

    def apply_catheter_load(self, load_grams: float, params):
        diameter = params['mechanical']['diameter']
        area = convert_diameter_to_area(diameter)
        force_N = convert_grams_to_newtons(load_grams)

        if area > 0: 
            pressure = force_N / area 
        else: 
            pressure = 0.0

        load_vec = np.zeros(self.dim)
        load_vec[-1] = -pressure
        self.T_hyp.value = load_vec

        if self.mesh.comm.rank == 0:
          print(
              f"Load = {load_grams:.3f} g, "
              f"F = {force_N:.6e} N, "
              f"d = {diameter:.6e} m, "
              f"A = {area:.6e} m^2, "
              f"traction = {pressure:.6e} Pa",
              flush=True
          )