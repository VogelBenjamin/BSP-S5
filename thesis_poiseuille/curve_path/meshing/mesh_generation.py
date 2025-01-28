from mpi4py import MPI
from petsc4py import PETSc
import numpy as np


from dolfinx.mesh import create_mesh, create_unit_square, create_rectangle, meshtags, locate_entities_boundary, CellType
from dolfinx.io import XDMFFile

# Curved pipe
Theta = np.pi/2


a = 1
b = 2
mesh = create_rectangle(
    MPI.COMM_WORLD,
    [np.array([a, 0.0]), np.array([b, 1.0])],
    [20, 20],
    cell_type=CellType.triangle
)


s = 1.3

# change distribution of nodes
def denser(x_arr):
    return  [[a + (b-a)*((x[0]-a)/(b-a))**s, x[1],x[2]] for x in x_arr]

x_bar = denser(mesh.geometry.x[:])
xy_bar_coor = np.array(x_bar)
mesh.geometry.x[:] = xy_bar_coor

# create curved pipe
def cylinder(x_arr):
    return [[x[0]*np.cos(Theta*x[1]), x[0]*np.sin(Theta*x[1]),x[2]] for x in x_arr]

x_hat = cylinder(x_bar[:])
xy_hat_coor = np.array(x_hat)
mesh.geometry.x[:] = xy_hat_coor
print(mesh.geometry.x)


# inlet region

mesh_in = create_rectangle(
    MPI.COMM_WORLD,
    [np.array([a, -1.0]), np.array([b, 0.0])],
    [20, 20],
    cell_type=CellType.triangle
)

# combine inlet to curved pipe
pipe_coor = mesh.geometry.x
inlet_coor = mesh_in.geometry.x

merged_coor = np.vstack([pipe_coor, inlet_coor])

pipe_cells = mesh.topology.connectivity(0,0).array.copy()
inlet_cells = mesh_in.topology.connectivity(0,0).array.copy()

offset_inlet = len(pipe_coor)

inlet_cells += offset_inlet

merged_cells = np.vstack([pipe_cells, inlet_cells])

from basix.ufl import element
pressure_fe = element("Lagrange", mesh.topology.cell_name(), 2,shape=(3,))

combined_mesh = create_mesh(MPI.COMM_WORLD, merged_cells, merged_coor,pressure_fe)



from pathlib import Path
from dolfinx.io import VTXWriter
folder = Path("mesh")
folder.mkdir(exist_ok=True, parents=True)
xdmf_filename = "mesh.xdmf"

with XDMFFile(MPI.COMM_WORLD, folder / xdmf_filename, "w") as xdmf_file:
    xdmf_file.write_mesh(combined_mesh)

