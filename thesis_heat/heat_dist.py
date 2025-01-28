import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx.mesh import create_interval
from dolfinx.fem import functionspace, dirichletbc, locate_dofs_geometrical, Function, Constant, assemble_scalar, form 
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile 


from ufl import TrialFunction, TestFunction, FacetNormal, ds, dx, grad, lhs, rhs, dot, inner 

k = 386
P = 2e-3
A_c = np.pi/4 * 1e-6
h = 20
Tw = 200
Tinf = 20
m = np.sqrt((h*P)/(k*A_c))


x0,x1 = 0, 1

mesh = create_interval(MPI.COMM_WORLD, 2, [x0,x1])


V = functionspace(mesh,('Lagrange', 2))

def wall_l(x):
    return np.isclose(x[0],x0)

wall_l_dot = locate_dofs_geometrical(V,wall_l)
heat_dbc = dirichletbc(PETSc.ScalarType(200), wall_l_dot, V)

u = TrialFunction(V)
v = TestFunction(V)

m_squared = Constant(mesh, PETSc.ScalarType(m*m))
Ti = Constant(mesh, PETSc.ScalarType(Tinf))

a = (dot(grad(u),grad(v)) + m_squared*dot(u,v))*dx 
L = m_squared*Ti*v*dx

problem = LinearProblem(a, L, bcs=[heat_dbc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()


length = x1-x0
cringe_const = np.cosh(m*length)

uD = Function(V)
uD.interpolate(lambda x: Tinf + ((Tw-Tinf)/(cringe_const))*np.cosh(m*(length-x[0])))


V2 = functionspace(mesh, ("Lagrange", 2))
uex = Function(V2)
uex.interpolate(lambda x: Tinf + ((Tw-Tinf)/(cringe_const))*np.cosh(m*(length-x[0])))
L2_error = form(inner(uh - uex, uh - uex) * dx)
error_local = assemble_scalar(L2_error)
error_L2 = np.sqrt(mesh.comm.allreduce(error_local, op=MPI.SUM))

error_max = np.max(np.abs(uD.x.array-uh.x.array))
# Only print the error on one process
if mesh.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")


bad = lambda x: Tinf + ((Tw-Tinf)/(cringe_const))*np.cosh(m*(length-x[0]))

if mesh.comm.rank == 0:  # Only print on rank 0 (master process)
    print(f"{'x':<15} {'u_approx':<15}")
    # Accessing the coordinates of the mesh nodes
    coords = mesh.geometry.x  # mesh.geometry.x contains the coordinates

    # Evaluate the function at each node
    for i in range(len(coords)):
        x_val = coords[i]  # Extract the x-coordinate (since it's a 1D mesh)
        u_val = uh.eval(np.array(x_val),True)  # Evaluate the solution at x_val
        u_true = bad(np.array(x_val))
        print(f"{x_val} {u_val[0]} {u_true}")


xdmf = XDMFFile(mesh.comm, "results/diffusion.xdmf", "w")
xdmf.write_mesh(mesh)
xdmf.write_function(uh,0)
