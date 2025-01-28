# BSP-S5
This repository stores all content regarding my Bachelor Semester Project for Semester 5 (2024/2025)

### Code execution

The easiest way to execute the programs is through docker (or singularity if you are on a HPC cluster)

##### Docker
```console
docker run -v <folder containing code>:/workspace -ti dolfinx/dolfinx:stable
cd /workspace
```
Navigate to program (example: n_curve_IPCS_vel.py)
```console
pyton3 n_curve_IPCS_vel.py
```


##### Singularity (on ULHPC)
```console
module load tools/Singularity
singularity pull docker://dolfinx/dolfinx:stable
```
Then either execute programs (example: n_curve_IPCS_vel.py) like this
```console
singularity exec dolfinx_latest.sif python3 n_curve_IPCS_vel.py
```
Or you can enter the environment using
```console
singularity run dolfinx_latest.sif 
```

##### Visualization
The results of the simulations are store in 'results' folders.
These folders contain .bp files for the velcoity and pressure of the system.
Loading them in paraview allows to visualize everything.

I recommend to:
- use a color gradient on the pressure data.
- set opacity of velocity to 0
- apply Glyph on velocity data + set orrientation to u_n + set color to magnitude of u_n
