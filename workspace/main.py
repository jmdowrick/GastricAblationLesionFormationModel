import numpy as np
from mpi4py import MPI
from dolfinx import fem, io
import basix
from datetime import datetime
import time

from src.electrostatics import ElectrostaticSolver
from src.bioheat import BioheatSolver
from src.celldeath import CellDeathSolver
from src.thermohyperelasticity import ThermoHyperElasticitySolver
from src.parameters import load_parameters

dimension = 2

# -----------
# Parameters
# -----------
params = load_parameters('parameters.yml')

t = params['simulation']['t']
dt = params['simulation']['dt']
t_end = params['simulation']['t_end']

target_power = params['electrical']['P_tar'] # Watts
initial_current = params['electrical']['I_0'] # Amps
power_tolerance = params['electrical']['P_tol']

mech_temp_trigger = params['mechanical']['temp_trigger']
mech_time_trigger = params['mechanical']['time_trigger']

comm = MPI.COMM_WORLD
save_output = True

if dimension == 2:
  path = "mesh/2D_axisymmetric/2D_axisymmetric.msh"

if dimension == 3:
  path = "mesh/tissue_only/tissue-mesh-catheter-facet.msh"

# ---------------------------------
# Initialise Domain and Solvers
# ---------------------------------
domain = io.gmsh.read_from_msh(path, comm, gdim=dimension)

# Scale from millimetres to metres 
domain.mesh.geometry.x[:,:] *=0.001 

# Define shared function spaces
V_shared = fem.Function(fem.functionspace(domain.mesh, ("Lagrange", 1))) # voltage
T_shared = fem.Function(fem.functionspace(domain.mesh, ("Lagrange", 1))) # temperature

# Deformation-gradient field
q_degree = 2
Ftot_element = basix.ufl.quadrature_element(cell=basix.CellType(dimension), value_shape=(3,3), degree=q_degree)
Ftot_shared = fem.Function(fem.functionspace(domain.mesh, Ftot_element))
Ftot_shared.interpolate(lambda x: np.tile(np.eye(3).flatten(), (x.shape[1], 1)).T)

# Output configurations
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_id = comm.bcast(run_id, root=0)
output_path_scalars = f'output/{run_id}_elec-heat-death-mech_scalars.bp'
output_path_disp = f'output/{run_id}_elec-heat-death-mech_disp.bp'

# Initialise solvers
cell_death_solver = CellDeathSolver(domain=domain,
                                    T_func=T_shared,
                                    params=params)

bio_solver = BioheatSolver(domain=domain,
                           V_func=V_shared,
                           T_func=T_shared,
                           Ftot_func=Ftot_shared,
                           params=params)

elec_solver = ElectrostaticSolver(domain=domain,
                                  T_func=T_shared,
                                  V_func=V_shared,
                                  Ftot_func=Ftot_shared,
                                  params=params,
                                  initial_current=initial_current)

mech_solver = ThermoHyperElasticitySolver(domain=domain,
                                          T_func=T_shared,
                                          CD_func=cell_death_solver.NUD,
                                          Ftot_func=Ftot_shared,
                                          params=params)

# Set unique names for functions for ParaView
V_shared.name = "Voltage"
T_shared.name = "Temperature"

# ------
# Pre-Ablation Prescribed Loading
# ------

load_steps = [10,15]
for load in load_steps:
    if comm.rank == 0:
        print(f" Applying pre-ablation load: {load} g", flush=True)

    mech_solver.apply_catheter_load(load, params)
    mech_solver.solve_step()

# Initial solve to establish baseline fields 
elec_solver.solve()
elec_solver.enforce_power_constraint(target_power, tol=power_tolerance) 
bio_solver.solve_step()

# Configure output
V_N, map_N = cell_death_solver.W.sub(0).collapse()
V_U, map_U = cell_death_solver.W.sub(1).collapse()
V_D, map_D = cell_death_solver.W.sub(2).collapse()
V_disp, map_disp = mech_solver.W.sub(0).collapse()

N_out = fem.Function(V_N)
N_out.name = "Healthy"
N_out.x.array[:] = cell_death_solver.NUD.x.array[map_N]

U_out = fem.Function(V_U)
U_out.name = "Damaged"
U_out.x.array[:] = cell_death_solver.NUD.x.array[map_U]

D_out = fem.Function(V_D)
D_out.name = "Dead"
D_out.x.array[:] = cell_death_solver.NUD.x.array[map_D]

u_out = fem.Function(V_disp)
u_out.name = "Displacement"
u_out.x.array[:] = mech_solver.up.x.array[map_disp] 

if save_output:
    vtx_scalars = io.VTXWriter(comm, output_path_scalars, [N_out, U_out, D_out, V_shared, T_shared], engine="BP4")
    vtx_disp = io.VTXWriter(comm, output_path_disp, [u_out], engine="BP4")
    vtx_scalars.write(t) # store initial conditions
    vtx_disp.write(t)

# ---------------------------------
# Time Stepping Loop
# ---------------------------------
tic = time.perf_counter()
output_interval = params['simulation']['output_interval']

next_save_time = t + output_interval
global_max = comm.allreduce(T_shared.x.array.max(), op=MPI.MAX)

t_prev_mech = t
T_max_prev = global_max

while t <= t_end and global_max < params['simulation']['temp_threshold']:
    t += dt
    
    # Update previous state variables
    bio_solver.advance_time()     
    cell_death_solver.advance_time()
    
    # Solve PDEs
    bio_solver.solve_step()
    cell_death_solver.solve_step()

    # Store cell states
    N_out.x.array[:] = cell_death_solver.NUD.x.array[map_N]
    U_out.x.array[:] = cell_death_solver.NUD.x.array[map_U]
    D_out.x.array[:] = cell_death_solver.NUD.x.array[map_D]
    
    # Check power constraint to see if electrical domain needs updating
    local_power = fem.assemble_scalar(elec_solver.power_form)
    current_power = comm.allreduce(local_power, op=MPI.SUM)

    if current_power < 1e-12:
        if comm.rank == 0:
            print("Power is effectively zero. Skipping controller adjustment.")
        continue
    
    lam = np.sqrt(current_power / target_power)
    
    if abs(lam - 1.0) < power_tolerance:
        pass
    else:
        # Update potential
        if comm.rank == 0:
            print(f" UPDATE potential, Ratio : {abs(lam - 1.0):.4f}", flush=True)
            print(f" Target/Current Dissipated Power: {target_power:.2f} , {current_power:.2f}", flush=True)
        
        # Scale the boundary condition voltage
        elec_solver.E_applied.value = elec_solver.E_applied.value / lam
        
        # Solve Electrostatic PDE
        elec_solver.solve()

    # Decide whether to update mechanics
    T_max_now = comm.allreduce(T_shared.x.array.max(), op=MPI.MAX)
    if (abs(T_max_now - T_max_prev) >= mech_temp_trigger) or ((t - t_prev_mech) > mech_time_trigger):
        if comm.rank == 0:
            print(" Solving mechanical problem", flush=True)

        mech_solver.solve_step()
        u_out.x.array[:] = mech_solver.up.x.array[map_disp] 
        T_max_prev = T_max_now
        t_prev_mech = t
    if comm.rank == 0:
        print(f"\nTime: {t:.1f} s", flush=True)
        
    if save_output and t >= (next_save_time - 1e-8): 
        vtx_scalars.write(t)
        vtx_disp.write(t)
        global_max = comm.allreduce(T_shared.x.array.max(), op=MPI.MAX)
        next_save_time += output_interval
        if comm.rank == 0:
            print(f"\nMax temp: {(global_max-273.15):.1f} oC", flush=True)

# --------
# Post-Ablation Load Removal
# --------
if comm.rank == 0:
    print("\nSolving mechanical problem after load removal (0 g)", flush = True)

mech_solver.apply_catheter_load(0.0, params)
mech_solver.solve_step()

u_out.x.array[:] = mech_solver.up.x.array[map_disp] 

if save_output:
    t_post = t + dt
    vtx_scalars.write(t)
    vtx_disp.write(t)
    vtx_scalars.close()
    vtx_disp.close()

toc = time.perf_counter()

if comm.rank == 0:
  print(f"Successfully ran the model in {toc - tic:0.1f} seconds")
