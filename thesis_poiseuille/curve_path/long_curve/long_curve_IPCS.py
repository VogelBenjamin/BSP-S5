'''
Workflow:
    1. Define the mesh
    2. Define the finite element spaces
    3. Define the boundary conditions
    4. Define the the functions/variables and constants
    5. Define the weak form of the problem
    6. Initialise and configure the solvers for the finite element method
    7. For each time step solve the problem and update the functions across all processes.
    8. (Optional) If the true solution is known, compare the approximation with the true solution 
       --> this is used to benchmark solution strategies.
    9. Save the outputs in a file format that can be processed by visualization software like paraview

'''

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
 
from dolfinx.mesh import create_unit_square, create_rectangle, meshtags, locate_entities_boundary, CellType
from basix.ufl import element
from dolfinx.fem import functionspace, locate_dofs_geometrical, locate_dofs_topological, dirichletbc, Constant, Function, form, assemble_scalar 
from ufl import (TrialFunction, TestFunction, FacetNormal, dx, dot, nabla_grad, nabla_div,
                 inner, lhs, rhs,sym,Identity, ds, div)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc



# Step 1

a = 2.0
b = 3.0
height = 2.0
mesh = create_rectangle(
    MPI.COMM_WORLD,
    [np.array([a, 0.0]), np.array([b, height])],
    [40, 20],
    cell_type=CellType.triangle
)

Theta = (np.pi/2) / height
s = 1.3

def denser(x_arr):
    return  [[a + (b-a)*((x[0]-a)/(b-a))**s, x[1],x[2]] for x in x_arr]

x_bar = denser(mesh.geometry.x[:])
xy_bar_coor = np.array(x_bar)
mesh.geometry.x[:] = xy_bar_coor
def cylinder(x_arr):
    return [[x[0]*np.cos(Theta*x[1]), x[0]*np.sin(Theta*x[1]),x[2]] for x in x_arr]

x_hat = cylinder(x_bar[:])
xy_hat_coor = np.array(x_hat)
mesh.geometry.x[:] = xy_hat_coor

'''
for el in mesh.geometry.x:
    if np.isclose(el[0],0):
        print(f"x=0: {el}")
    if np.isclose(el[1],0):
        print(f"y=0: {el}")
'''
# Mark boundaries before transformation 
facet_indices, facet_markers = [], []

'''
# WALL BOUNDARY CONITIONS ARE FALSE
'''
def is_wall(x):
    return np.logical_or(np.isclose(np.sqrt(x[0]**2 + x[1]**2), a),np.isclose(np.sqrt(x[0]**2 + x[1]**2), b))

def is_inlet(x):
    return np.isclose(x[1], 0)

def is_outlet(x):
    return np.isclose(x[0], 0)


for marker, locator in [(1, is_wall), (2, is_inlet), (3, is_outlet)]:
    facets = locate_entities_boundary(mesh, 1, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))

facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)


boundaries = meshtags(mesh, 1, facet_indices[sorted_facets], facet_markers[sorted_facets])

#Step 2
# arguments: type of interpolation, geometry of fininte element, order of element, output dimension 
velocity_fe = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim,))
velocity_fem_space = functionspace(mesh,velocity_fe)

pressure_fe = element("Lagrange", mesh.topology.cell_name(), 1)
pressure_fem_space = functionspace(mesh, pressure_fe)

# Step 3
# define functions that return boolean based on whether a region belongs to the boundary space

walls_dof = locate_dofs_topological(velocity_fem_space, 1, boundaries.find(1))
inlet_dof = locate_dofs_topological(pressure_fem_space, 1, boundaries.find(2))
outlet_dof = locate_dofs_topological(pressure_fem_space, 1, boundaries.find(3))

# define the values the nodes should take at the boundaries
u_noslip = np.zeros(mesh.geometry.dim).astype(PETSc.ScalarType) # bc at walls
p_inlet = PETSc.ScalarType(4) # pressure at inlet
p_outlet = PETSc.ScalarType(0) # pressure at outlet

# initalise boundary conditions

walls_bc = dirichletbc(u_noslip, walls_dof, velocity_fem_space)
inlet_bc = dirichletbc(p_inlet, inlet_dof, pressure_fem_space)
outlet_bc = dirichletbc(p_outlet, outlet_dof, pressure_fem_space)

u_bc = [walls_bc]
p_bc = [inlet_bc,outlet_bc]

# Step 4

# time variable
T = 10 # simulation time
time_steps = 1000

# constants
f = Constant(mesh, PETSc.ScalarType((0,0)))
rho = Constant(mesh, PETSc.ScalarType(1))
mu = Constant(mesh, PETSc.ScalarType(1))
dt = Constant(mesh, PETSc.ScalarType(T/time_steps))

# unknowns
u = TrialFunction(velocity_fem_space)
p = TrialFunction(pressure_fem_space)

# test functions (weighing function)
v = TestFunction(velocity_fem_space)
q = TestFunction(pressure_fem_space)

# functions
u_n = Function(velocity_fem_space)
u_n.name = "u_n"
U = (u + u_n) * 0.5
u_ = Function(velocity_fem_space)
u_.name = "u_"
p_n = Function(pressure_fem_space)
p_n.name= "p_n"
p_ = Function(pressure_fem_space)
p_.name= "p_"

n = FacetNormal(mesh)

def epsilon(u):
    return sym(nabla_grad(u))

def sigma(u,p):
    return 2 * mu * epsilon(u) - p * Identity(len(u))


# Step 5
# tentative velocity

'''
tv_problem = (1/dt)*inner(u - u_n, v)*dx 
tv_problem += inner(dot(u_n, nabla_grad(u_n)), v)*dx 
tv_problem += (1/rho)*inner(sigma(u_half,p_n_1), epsilon(v))*dx
tv_problem += (1/rho)*inner(p_n_1*n,v)*ds
tv_problem += -(mu/rho)*inner(dot(nabla_grad(u_half).T,n),v)*ds
tv_problem -= inner(f, v)*dx
'''

tv_problem = rho * dot((u - u_n) / dt, v) * dx
tv_problem += rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx
tv_problem += inner(sigma(U, p_n), epsilon(v)) * dx
tv_problem += dot(p_n * n, v) * ds - dot(mu * nabla_grad(U) * n, v) * ds
tv_problem -= dot(f, v) * dx


# pressure

'''
p_problem = inner(nabla_grad(p), nabla_grad(q))*dx
p_problem -= -(rho/dt)*inner(nabla_div(u_star),q)*dx
p_problem -= (rho/dt)*inner(nabla_grad(p_n_1),nabla_grad(q))*dx
'''

p_problem = dot(nabla_grad(p), nabla_grad(q)) * dx
p_problem -= dot(nabla_grad(p_n), nabla_grad(q)) * dx - (rho / dt) * div(u_) * q * dx

# velocity update
vu_problem = rho * dot(u, v) * dx
vu_problem -= rho * dot(u_, v) * dx - dt * dot(nabla_grad(p_ - p_n), v) * dx 


# Step 6
# tentative velocity solver
tv_bilinear = form(lhs(tv_problem)) # contains all elements containing trialfunction
tv_linear = form(rhs(tv_problem)) # contains all elements without trial function

tv_A = assemble_matrix(tv_bilinear, bcs=u_bc)
tv_A.assemble()
tv_b = create_vector(tv_linear)

tv_solver = PETSc.KSP().create(mesh.comm)
tv_solver.setOperators(tv_A)
tv_solver.setType(PETSc.KSP.Type.BCGS)
tv_precond = tv_solver.getPC()
tv_precond.setType(PETSc.PC.Type.HYPRE)
tv_precond.setHYPREType("boomeramg")

# pressure 
p_bilinear = form(lhs(p_problem)) #form(lhs(p_problem)) # contains all elements containing trialfunction
p_linear = form(rhs(p_problem)) #form(rhs(p_problem)) # contains all elements without trial function

p_A = assemble_matrix(p_bilinear, bcs=p_bc)
p_A.assemble()
p_b = create_vector(p_linear)

p_solver = PETSc.KSP().create(mesh.comm)
p_solver.setOperators(p_A)
p_solver.setType(PETSc.KSP.Type.BCGS)
p_precond = p_solver.getPC()
p_precond.setType(PETSc.PC.Type.HYPRE)
p_precond.setHYPREType("boomeramg")

# velocity update
vu_bilinear = form(lhs(vu_problem)) #form(lhs(vu_problem)) # contains all elements containing trialfunction
vu_linear = form(rhs(vu_problem)) #form(rhs(vu_problem)) # contains all elements without trial function

vu_A = assemble_matrix(vu_bilinear, bcs=u_bc)
vu_A.assemble()
vu_b = create_vector(vu_linear)

vu_solver = PETSc.KSP().create(mesh.comm)
vu_solver.setOperators(vu_A)
vu_solver.setType(PETSc.KSP.Type.CG)
vu_precond = vu_solver.getPC()
vu_precond.setType(PETSc.PC.Type.SOR)

# step 7, 8, 9

from pathlib import Path
from dolfinx.io import VTXWriter
folder = Path("long_results")
folder.mkdir(exist_ok=True, parents=True)
vtx_u = VTXWriter(mesh.comm, folder / "poiseuille_u.bp", u_n, engine="BP4")
vtx_p = VTXWriter(mesh.comm, folder / "poiseuille_p.bp", p_n, engine="BP4")


def u_exact(x):
    values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
    values[0] = 4 * x[1] * (1.0 - x[1])
    return values


u_ex = Function(velocity_fem_space)
u_ex.interpolate(u_exact)

u_final = Function(velocity_fem_space)
L2_error = form(dot(u_n - u_ex, u_n - u_ex) * dx)

i = 0
for t in np.linspace(0,T,time_steps):
    

    # tentative velocity
    with tv_b.localForm() as loc:
        loc.set(0) # reset the constant vector to 0
    assemble_vector(tv_b,tv_linear) # construct constant vector of the global system
    apply_lifting(tv_b, [tv_bilinear], [u_bc]) # apply boundary conditions to the vector

    # make sure that neighbouring nodes exchange information that might be needed during computations
    # to guarantee consistency
    tv_b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE) 
    set_bc(tv_b, u_bc) # enforces the boundary condition
    tv_solver.solve(tv_b, u_.x.petsc_vec)
    u_.x.scatter_forward()

    # pressure
    with p_b.localForm() as loc:
        loc.set(0)
    assemble_vector(p_b, p_linear)
    apply_lifting(p_b, [p_bilinear], [p_bc])

    p_b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(p_b,p_bc)
    p_solver.solve(p_b, p_.x.petsc_vec)
    p_.x.scatter_forward()
    
    # vector update
    with vu_b.localForm() as loc:
        loc.set(0)
    assemble_vector(vu_b, vu_linear)
    apply_lifting(vu_b, [vu_bilinear], [u_bc])

    vu_b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(vu_b,u_bc)
    vu_solver.solve(vu_b, u_.x.petsc_vec)
    u_.x.scatter_forward()

    u_n.x.array[:] = u_.x.array[:]
    p_n.x.array[:] = p_.x.array[:]
    
    # Compute error at current time-step
    error_L2 = np.sqrt(mesh.comm.allreduce(assemble_scalar(L2_error), op=MPI.SUM))
    error_max = mesh.comm.allreduce(np.max(u_n.x.petsc_vec.array - u_ex.x.petsc_vec.array), op=MPI.MAX)
    # Print error only every 20th step and at the last step
    if (i % 100 == 0) or (i == time_steps - 1):
        divergence = np.sqrt(mesh.comm.allreduce(assemble_scalar(form(nabla_div(u_n) * nabla_div(u_n) * dx)), op=MPI.SUM))
        print(f"Time {t:.2f}, L2-error {error_L2:.2e}, Max error {error_max:.2e}", "Divergence:", divergence)
        vtx_u.write(t)
        vtx_p.write(t)
    i += 1

vtx_u.close()
vtx_p.close()

tv_b.destroy()
p_b.destroy()
vu_b.destroy()

tv_solver.destroy()
p_solver.destroy()
vu_solver.destroy()
