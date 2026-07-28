import numpy as np
from mpi4py import MPI
from dolfinx import fem, io
from datetime import datetime
import time

from src.electrostatics import ElectrostaticSolver
from src.bioheat import BioheatSolver
from src.celldeath import CellDeathSolver
from src.parameters import load_parameters

# Parameters
params = load_parameters('parameters.yml')

t = params['simulation']['t']
dt = params['simulation']['dt'] 
t_end = params['simulation']['t_end']

target_power = params['electrical']['P_tar'] # Watts
initial_current = params['electrical']['I_0'] # Amps
power_tolerance = params['electrical']['P_tol']

comm = MPI.COMM_WORLD
save_output = True

# ---------------------------------
# Initialize Domain and Solvers
# ---------------------------------
domain = io.gmsh.read_from_msh("mesh/tissue_only/tissue-mesh-catheter-facet.msh", comm)

# Scale from millimetres to metres 
domain.mesh.geometry.x[:,:] *=0.001 

# Define shared function spaces
V_shared = fem.Function(fem.functionspace(domain.mesh, ("Lagrange", 1)))
T_shared = fem.Function(fem.functionspace(domain.mesh, ("Lagrange", 1)))

# Output configurations
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_id = comm.bcast(run_id, root=0)
output_path = f'output/{run_id}_elec-heat.bp'

# Initialise solvers
bio_solver = BioheatSolver(domain=domain,
                           V_func=V_shared,
                           T_func=T_shared,
                           params=params)

elec_solver = ElectrostaticSolver(domain=domain,
                                  T_func=T_shared,
                                  V_func=V_shared,
                                  params=params,
                                  initial_current=initial_current)

cell_death_solver = CellDeathSolver(domain=domain,
                                    T_func=T_shared,
                                    params=params)

# Set unique names for functions for ParaView
V_shared.name = "Voltage"
T_shared.name = "Temperature"

# Initial solve to establish baseline fields 
elec_solver.solve()
elec_solver.enforce_power_constraint(target_power, tol=power_tolerance) 
bio_solver.solve_step()

# Configure output
V_N, map_N = cell_death_solver.W.sub(0).collapse()
V_U, map_U = cell_death_solver.W.sub(1).collapse()
V_D, map_D = cell_death_solver.W.sub(2).collapse()

N_out = fem.Function(V_N)
N_out.name = "Healthy"
N_out.x.array[:] = cell_death_solver.NUD.x.array[map_N]

U_out = fem.Function(V_U)
U_out.name = "Damaged"
U_out.x.array[:] = cell_death_solver.NUD.x.array[map_U]

D_out = fem.Function(V_D)
D_out.name = "Dead"
D_out.x.array[:] = cell_death_solver.NUD.x.array[map_D]

if save_output:
    vtx = io.VTXWriter(comm, output_path, [N_out, U_out, D_out, V_shared, T_shared], engine="BP4")
    vtx.write(t) # store initial conditions

# ---------------------------------
# Time Stepping Loop
# ---------------------------------
tic = time.perf_counter()
output_interval = params['simulation']['output_interval']

next_save_time = t + output_interval
global_max = comm.allreduce(T_shared.x.array.max(), op=MPI.MAX)

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

    if comm.rank == 0:
        print(f"\nTime: {t:.1f} s", flush=True)
        
    if save_output and t >= (next_save_time - 1e-8): 
        vtx.write(t)
        global_max = comm.allreduce(T_shared.x.array.max(), op=MPI.MAX)
        next_save_time += output_interval
        if comm.rank == 0:
            print(f"\nMax temp: {(global_max-273.15):.1f} oC", flush=True)

if save_output:
    vtx.close()

toc = time.perf_counter()

if comm.rank == 0:
  print(f"Successfully ran the model in {toc - tic:0.1f} seconds")
