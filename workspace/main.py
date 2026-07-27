import numpy as np
from mpi4py import MPI
from dolfinx import fem, io
from datetime import datetime
import time
import ufl

from src.electrostatics import ElectrostaticSolver
from src.bioheat import BioheatSolver
from src.parameters import load_parameters

# Parameters
t = 0.0
dt = 0.1
t_end = 5.0
target_power = 15.0 # Watts
initial_current = 1000 # Amps
power_tolerance = 0.01
params = load_parameters('parameters.yml')
comm = MPI.COMM_WORLD
save_output = False

# ---------------------------------
# Initialize Domain and Solvers
# ---------------------------------
domain = io.gmsh.read_from_msh("mesh/3D_ablation/catheter-in-air-on-tissue.msh", comm)

# Scale from millimetres to metres 
domain.mesh.geometry.x[:,:] *=0.001

# Find boundaries
electrode_tags = domain.physical_groups['catheter_bottom'].tag
ds_electrode = ufl.Measure("dS", domain=domain.mesh, subdomain_data=domain.facet_tags) 
ground_facets = domain.facet_tags.find(domain.physical_groups['base'].tag)

# Define shared function space
V_shared = fem.Function(fem.functionspace(domain.mesh, ("Lagrange", 1)))

# Output configurations
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_id = comm.bcast(run_id, root=0)
output_path = f'output/{run_id}_elec-heat.bp'

# Initialise solvers
bio_solver = BioheatSolver(mesh=domain.mesh,
                           V_func=V_shared,
                           dt=dt,
                           params=params)

elec_solver = ElectrostaticSolver(mesh=domain.mesh,
                                  T_func=bio_solver.T,
                                  V_func=V_shared,
                                  params=params,
                                  electrode_tags=electrode_tags,
                                  ds_electrode=ds_electrode,
                                  ground_facets=ground_facets,
                                  initial_current=initial_current)

# Set unique names for functions for ParaView
elec_solver.V.name = "Voltage"
bio_solver.T.name = "Temperature"

# Initial solve to establish baseline fields 
elec_solver.solve()
elec_solver.enforce_power_constraint(target_power, tol=power_tolerance) 
bio_solver.solve_step()

# Configure output
if save_output:
    vtx = io.VTXWriter(comm, output_path, [elec_solver.V, bio_solver.T], engine="BP4")

# ---------------------------------
# Time Stepping Loop
# ---------------------------------
tic = time.perf_counter()
while t <= t_end:
    t += dt
    
    # A. Update previous state variables
    bio_solver.T_n.x.array[:] = bio_solver.T.x.array
    
    # B. Solve Bioheat PDE
    # Uses the current electric field and the previous temperature
    bio_solver.solve_step()
    
    # C. Check Power Constraint
    # Calculate current power using the new temperature distribution
    local_power = fem.assemble_scalar(elec_solver.power_form)
    current_power = comm.allreduce(local_power, op=MPI.SUM)
    
    lam = np.sqrt(current_power / target_power)
    
    if abs(lam - 1.0) < power_tolerance:
        pass
    else:
        # Update potential
        if comm.rank == 0:
            print(f" UPDATE potential, Ratio : {abs(lam - 1.0):.4f}")
            print(f" Target/Current Dissipated Power: {target_power:.2f} , {current_power:.2f}")
        
        # Scale the boundary condition voltage
        elec_solver.E_applied.value = elec_solver.E_applied.value / lam
        
        # D. Solve Electrostatic PDE
        elec_solver.solve()


    if comm.rank == 0:
        if save_output: 
            vtx.write(t)
        print(f"\nTime: {t:.1f} s")

if save_output:
    vtx.close()

toc = time.perf_counter()
print(f"Successfully ran the model in {toc - tic:0.1f} seconds")
