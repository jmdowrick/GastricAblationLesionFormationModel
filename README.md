# Modelling lesion formation following gastric serosal RFA

A finite element model of lesion formation in stomach tissue following radio-frequency ablation on the serosal surface. Contains loosely coupled electrical and thermal domains, with a three-state cell-state ODE model that tracks healthy, damaged, and dead cells. Will include mechanical deformation arising from catheter tip. Based on [Molinari et al. (2021)](https://doi.org/10.1016/j.jmps.2022.104810).

## Pre-requisites
1. [Visual Studio Code](https://code.visualstudio.com/) with [Container Tools](https://code.visualstudio.com/docs/containers/overview) extension
2. [Docker](https://docs.docker.com/desktop/setup/install/mac-install/)
3. Might be something with MPI - to confirm

## Start guide
1. Clone the repository 

```
cd [working_dir]
git clone https://github.com/TargetLaboratory/GastricAblationLesionFormationModel
cd GastricAblationLesionFormationModel
```

2. Open in visual studio and start container using `ctrl + shift + P`

3. Run in parallel using the following command in the terminal

```
mpiexec -n 4 python3 main.py
```

4. `.bp` (`.vtk`) outputs will be stored in the `./workspace/output` directory and can be visualised using [Paraview](https://www.paraview.org/)

## Model outputs
- Voltage (V) 
- Temperature (K)
- Healthy 
- Damaged
- Dead

## Project structure
