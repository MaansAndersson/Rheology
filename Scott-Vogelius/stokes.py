
from mshr import *
from dolfin import *
import math,sys
from scipy.io import loadmat, savemat
from timeit import default_timer as timer

set_log_active(True)

mesh = Mesh("mesh.xml")
dim = mesh.geometric_dimension()

cell = mesh.ufl_cell()

pdeg = 4
print("pdeg: ", pdeg)
fudg = 10000
print('Diensions: ', dim)


#int(sys.argv[0])
#meshsize= 24 #int(sys.argv[1])
design = "classic" #str(sys.argv[2])
lbufr = -1; #float(sys.argv[3])
rbufr = 3; #float(sys.argv[4])
r0 = 0.5; #float(sys.argv[4])
r1 = 1; #float(sys.argv[4])
upright = 1.0 #0.5
right = 1.0

h = CellDiameter(mesh)

vtkfile_stokes_U = File('results/ust.pvd')
vtkfile_stokes_P = File('results/pst.pvd')
vtkfile_stokes_Uxml = File('ust.xml')
#vtkfile_stokes_Pxml = File('pst.xml')

#V1 = FiniteElement("Lagrange", mesh.ufl_cell(), pdeg)
#B = FiniteElement("B", mesh.ufl_cell(), mesh.topology().dim() + 1)
#V = FunctionSpace(mesh, VectorElement(NodalEnrichedElement(V1, B)))
V = VectorFunctionSpace(mesh, "Lagrange", pdeg)

Q = FunctionSpace(mesh, "Lagrange", pdeg-1)

# define boundary condition
if dim == 2 :
    boundary_exp = Expression(("exp(-fu*(lb-x[0])*(lb-x[0]))*(1.0-x[1]*x[1]) + \
      (1.0/up)*exp(-fu*(ri+rb-x[0])*(ri+rb-x[0]))*(1.0-((x[1]*x[1])/(up*up)))","0"), \
                       up=upright,ri=right,fu=fudg,rb=rbufr,lb=lbufr,degree = pdeg)
else : 
    boundary_exp = Expression(("exp(-fu*(lb-x[0])*(lb-x[0]))*(1.0-(x[1]*x[1]+x[2]*x[2])) + \
      (1.0/up)*exp(-fu*(ri+rb-x[0])*(ri+rb-x[0]))*(1.0-((x[1]*x[1]+x[2]*x[2])/(up*up)))","0","0"), \
                       up=upright,ri=right,fu=fudg,rb=rbufr,lb=lbufr,degree = pdeg)
                       

# Stoke's
bc = DirichletBC(V, boundary_exp, "on_boundary")

# set the parameters
r = 1.0e4

# define test and trial functions, and function that is updated
uold = TrialFunction(V)
v = TestFunction(V)
w = Function(V)
w.vector()[:] = 0

asf = inner(grad(uold),grad(v))*dx + r*div(uold)*div(v)*dx
bs = -div(w)*div(v)*dx 

uold = Function(V)
ust = Function(V)

#pdes = LinearVariationalProblem(asf, bs, uold, bc)
#solvers = LinearVariationalSolver(pdes)
# Stokes solution
# Scott-Vogelius iterated penalty method
iters = 0; max_iters = 10; div_u_norm = 1
while iters < max_iters and div_u_norm > 1e-10:
# solve and update w
    start = timer()
    A = assemble(asf)
    b = assemble(bs)
    end = timer()
    print('Assemble time: ', end - start)
    
    start = timer()
    bc.apply(A)
    bc.apply(b)
    end = timer()
    print('Apply BC time: ', end - start)
    
    start = timer()
    solve(A, uold.vector(), b, 'tfqmr') #, 'gmres','amg')
    end = timer()
    print('Linear solver time: ', end - start)


    
    #solvers.solve()
    w.vector().axpy(r, uold.vector())
# find the L^2 norm of div(u) to check stopping condition
    div_u_norm = sqrt(assemble(div(uold)*div(uold)*dx(mesh)))
    if(MPI.rank(mesh.mpi_comm()) == 0):
        print( "   IPM iter_no=",iters,"div_u_norm="," %.5e"%div_u_norm)
    iters += 1
if(MPI.rank(mesh.mpi_comm()) == 0):
    print( "Stokes solver  IPM iter_no=",iters,"div_u_norm="," %.2e"%div_u_norm)
#pstoke = calculate_pressure_drop(-div(w)/reno, V, mesh)
#fstoke, fpstoke, ftstoke = calculate_total_force(uold,-1/reno*div(w), mesh)

uold.set_allow_extrapolation(True)


ust.vector().axpy(1.0, uold.vector())
vtkfile_stokes_U << project(uold,V)
vtkfile_stokes_P << project(-div(w),Q)



#Not MPI-safe.
if False:
    uvec = uold.vector()[:].reshape(len(uold.vector()),1)
    savemat('ust', { 'uvec': uvec }, oned_as='column')

vtkfile_stokes_Uxml << project(uold,V)
#vtkfile_stokes_Pxml << project(-div(w),Q)
