
from mshr import *
from dolfin import *
import math,sys
from scipy.io import loadmat, savemat
set_log_active(False)

pdeg = 4  #6
fudg = 10000

reno = 1.


lbufr = -1; #float(sys.argv[3])
rbufr = 3; #float(sys.argv[4])
r0 = 0.5; #float(sys.argv[4])
r1 = 1; #float(sys.argv[4])
upright = 1. #0.5
right = 1.

mesh = Mesh("mesh.xml")
dim = mesh.geometric_dimension()

vtkfile_navierstokes_U = File('results/unst.pvd')
vtkfile_navierstokes_P = File('results/pnst.pvd')
vtkfile_navierstokes_dU = File('results/dunst.pvd')

V = VectorFunctionSpace(mesh, "Lagrange", pdeg)
Q = FunctionSpace(mesh, "Lagrange", pdeg-1)

# define boundary condition
if dim == 2 :
    boundary_exp = Expression(("exp(-fu*(lb-x[0])*(lb-x[0]))*(1.0-x[1]*x[1]) + \
      (1.0/up)*exp(-fu*(ri+rb-x[0])*(ri+rb-x[0]))*(1.0-((x[1]*x[1])/(up*up)))","0"), \
                       up=upright,ri=right,fu=fudg,rb=rbufr,lb=lbufr,degree = pdeg)
                       
    zf = Expression(("0","0"), degree = pdeg)
else :
    boundary_exp = Expression(("exp(-fu*(lb-x[0])*(lb-x[0]))*(1.0-(x[1]*x[1]+x[2]*x[2])) + \
      (1.0/up)*exp(-fu*(ri+rb-x[0])*(ri+rb-x[0]))*(1.0-((x[1]*x[1]+x[2]*x[2])/(up*up)))","0","0"), \
                       up=upright,ri=right,fu=fudg,rb=rbufr,lb=lbufr,degree = pdeg)
    
    zf = Expression(("0","0","0"), degree = pdeg)


# Navier-Stoke
bc = DirichletBC(V, boundary_exp, "on_boundary")
bcz = DirichletBC(V, zf, "on_boundary")

# set the parameters
r = 1.0e4

#
uold = Function(V)
uold.vector()[:] = 0.0

ust = Function(V)
uvec_old = loadmat('ust')['uvec']
ust.vector().set_local(uvec_old[:,0])
uold.vector().axpy(1, ust.vector())

kters = 0; max_kters = 9; unorm = 1
while kters < max_kters and unorm > 1e-6:
    
    u = TrialFunction(V)
    v = TestFunction(V)
    w = Function(V)
    w.vector()[:] = 0

    a = inner(grad(u), grad(v))*dx + r*div(u)*div(v)*dx \
      +reno*inner(grad(uold)*u,v)*dx+reno*inner(grad(u)*uold,v)*dx
    b = -div(w)*div(v)*dx
    F = inner(grad(uold), grad(v))*dx+reno*inner(grad(uold)*uold,v)*dx
    
    u = Function(V)
    pde = LinearVariationalProblem(a, F + b, u, bcz)
    solver = LinearVariationalSolver(pde)
    
    # Scott-Vogelius iterated penalty method
    iters = 0; max_iters = 10; div_u_norm = 1

#   iters = 0; max_iters = max(2,kters); div_u_norm = 1
    while iters < max_iters and div_u_norm > 1e-10:
    # solve and update w
        solver.solve()
        w.vector().axpy(r, u.vector())
        # find the L^2 norm of div(u) to check stopping condition
        div_u_norm = sqrt(assemble(div(u)*div(u)*dx(mesh)))
        if(MPI.rank(mesh.mpi_comm()) == 0):
            print( "   IPM iter_no=",iters,"div_u_norm="," %.2e"%div_u_norm)
        iters += 1
    if(MPI.rank(mesh.mpi_comm()) == 0):
        print( "   IPM iter_no=",iters,"div_u_norm="," %.2e"%div_u_norm)
    kters += 1
    uold.vector().axpy(-1.0, u.vector())
    unorm=norm(u,norm_type='H1')
    uoldnorm=norm(uold,norm_type='H1')
    div_u_norm = sqrt(assemble(div(uold)*div(uold)*dx(mesh)))
    if(MPI.rank(mesh.mpi_comm()) == 0):
        print( "Newton iter_no=",kters,"Delta_u_norm="," %.2e"%unorm,"u_norm=", \
          " %.2e"%uoldnorm,"div_u_norm="," %.2e"%div_u_norm)

# Stoke's projection
uz = TrialFunction(V)
v = TestFunction(V)
w.vector()[:] = 0

ap = inner(grad(uz), grad(v))*dx + r*div(uz)*div(v)*dx
b = -div(w)*div(v)*dx
Fp = -reno*inner(grad(uold)*uold,v)*dx

iter = 0
div_u_norm = 1
uz = Function(V)
pdep = LinearVariationalProblem(ap, Fp + b, uz, bc)
while(iter < max_iters and div_u_norm > 1e-10):
    solvep = LinearVariationalSolver(pdep)
    solvep.solve()
    w.vector().axpy(r, uz.vector())
    iter += 1
    div_u_norm = sqrt(assemble(div(uz)*div(uz)*dx(mesh)))
#pnse = calculate_pressure_drop(-1/reno*div(w), V, mesh)
#fnse, fpnse, ftnse = calculate_total_force(uz,-1/reno*div(w),mesh)
vtkfile_navierstokes_U << uz
vtkfile_navierstokes_P << project(-div(w)/reno,Q)

#plot(uold[0], interactive=True)
nst = Function(V)
# nst is the Navier-Stokes solution
nst.vector().axpy(1.0, uz.vector())

uvec = uz.vector()[:].reshape(len(uz.vector()),1)
savemat('nst', { 'uvec': uvec }, oned_as='column')

vtkfile_navierstokes_U << project(nst,V)
vtkfile_navierstokes_P << project(-div(w)/reno,Q)
nst.vector().axpy(-1,ust.vector())
vtkfile_navierstokes_dU << project(nst,V)


