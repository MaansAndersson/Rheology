from mshr import *
from dolfin import *
import math,sys
from scipy.io import loadmat, savemat
from ufl import nabla_div
#set_log_active(False)

pdeg = 3
fudg = 10000
reno = 1.
alfa = 0.01
lbufr = -1; #float(sys.argv[3])
rbufr = 3; #float(sys.argv[4])
r0 = 0.5; #float(sys.argv[4])
r1 = 1; #float(sys.argv[4])
upright = 0.5
right = 1.0 #0.5

alpha_1 = Constant(alfa)
alpha_2 = Constant(-alfa)

mesh = Mesh("mesh.xml")
dim = mesh.geometric_dimension()
print(dim)

vtkfile_navierstokes_U = File('results/g2uW.pvd')
vtkfile_navierstokes_P = File('results/g2pW.pvd')
vtkfile_navierstokes_dU = File('results/grade2W-nse.pvd')

V = VectorFunctionSpace(mesh, "Lagrange", pdeg)
Q = FunctionSpace(mesh, "Lagrange", pdeg-1)
V2 = VectorFunctionSpace(mesh, "CG", pdeg)
#Y = TensorFunctionSpace(mesh, "CG", 1, shape = (dim,dim))

def in_bdry(x):
    return x[0] <= lbufr+DOLFIN_EPS

# define boundary condition
if dim == 2 :
    boundary_exp = Expression(("exp(-fu*(lb-x[0])*(lb-x[0]))*(1.0-x[1]*x[1]) + \
      (1.0/up)*exp(-fu*(ri+rb-x[0])*(ri+rb-x[0]))*(1.0-((x[1]*x[1])/(up*up)))","0"), \
                       up=upright,ri=right,fu=fudg,rb=rbufr,lb=lbufr,degree = pdeg)
                       
    WINFLOW = Expression(("0","8*x[1]*(3*a1+2*a2)"), a1 = alpha_1, a2 = alpha_2, degree = pdeg)


else :
    boundary_exp = Expression(("exp(-fu*(lb-x[0])*(lb-x[0]))*(1.0-(x[1]*x[1]+x[2]*x[2])) + \
      (1.0/up)*exp(-fu*(ri+rb-x[0])*(ri+rb-x[0]))*(1.0-((x[1]*x[1]+x[2]*x[2])/(up*up)))","0","0"), \
                       up=upright,ri=right,fu=fudg,rb=rbufr,lb=lbufr,degree = pdeg)
    

bc = DirichletBC(V, boundary_exp, "on_boundary")
bcW =  DirichletBC(V2, WINFLOW,  in_bdry, 'pointwise')

# set the parameters
r = 1.0e4


W = Function(V2)
q = Function(Q)
u = TrialFunction(V)
v = TestFunction(V)
v2 = TestFunction(V2)
Uoldr = Function(V)
U = Function(V)

nst = Function(V)
uvec_old = loadmat('nst')['uvec']
nst.vector().set_local(uvec_old[:,0])
U.vector().axpy(1, nst.vector())
#inintial guess
#U.vector()[:] = 1 # Should be NSE.
q.vector()[:] = 0


h = CellDiameter(mesh)
gg2_iter = -1
max_gg2_iter = 10
max_piter = 5

#Pre-define variables
incrnorm = 1; gtol = alfa*0.0001;  r = 1.0e4;
n = FacetNormal(mesh)

def A(z):
    return (grad(z) + grad(z).T)

wgg2 = Function(V)
while gg2_iter < max_gg2_iter and incrnorm > gtol:
   wgg2.vector()[:] = 0

   piter = 0;  div_u_norm = 1
   gg2_iter += 1
   
   # FORMS FOR STRESS
   rz =  inner(W \
     + alpha_1*dot(U, nabla_grad(W)) \
     - div(alpha_1*grad(U).T*A(U) \
     + (alpha_1 + alpha_2)*A(U)*A(U) \
     - outer(U,U) \
     - alpha_1*q*grad(U).T), v2)*dx(mesh) \
     
  # + abs(alpha_1*dot(U('+'),n('+')))*conditional(dot(U('+'),n('+'))<0,1,0)*inner(jump(W),v2('+'))*dS(mesh)

   print('solving for stress')
   solve(rz == 0, W, bcW, solver_parameters={"newton_solver": {"relative_tolerance": 1e-11}})
   
   # FORMS FOR IPM
   a_gg2 = inner(grad(u), grad(v))*dx(mesh) + r*div(u)*div(v)*dx(mesh)
   F_gg2 = inner(W,v)*dx(mesh)
   b_gg2 = -div(wgg2)*div(v)*dx(mesh)
   
   
   # Solver IPM
   pdegg2 = LinearVariationalProblem(a_gg2, F_gg2 + b_gg2, U, bc)
   solvergg2 = LinearVariationalSolver(pdegg2)


   while (piter < max_piter and div_u_norm > 1.0E-10):
       solvergg2.solve()
       wgg2.vector().axpy(r, U.vector())
       # Criteria for stopping

       div_u_norm = sqrt(assemble(div(U)*div(U)*dx(mesh)))
       piter += 1
       
       #Uoldr.vector().axpy(-1, U.vector())
       #incrnorm = norm(Uoldr,'H1')
       #incrnorm /= norm(U,'H1')
       
       #Uoldr.vector()[:] = 0
       #Uoldr.vector().axpy(1, U.vector())
       incrnorm = errornorm(U,Uoldr,norm_type='H1',degree_rise=0)/norm(U,norm_type='H1')
       if(MPI.rank(mesh.mpi_comm()) == 0):
            print("div(u) ",piter , div_u_norm)
    
   assign(q, project(-(div(wgg2)),Q))
   if(MPI.rank(mesh.mpi_comm()) == 0):
       print( "   IPM iter_no=",piter,"div_u_norm="," %.2e"%div_u_norm)
       print( gg2_iter,"change="," %.3e"%incrnorm)
   #print(errornorm(SIGMA,sigma0,norm_type='H1',degree_rise=2)/norm(SIGMA,norm_type='H1'))
   

   Uoldr = Function(V)
   Uoldr.vector().axpy(1, U.vector())

   G2D = Function(V)
   G2D.vector().axpy(1, U.vector())
    #assign(G2D,U)

   vtkfile_navierstokes_U << project(U,V)
   vtkfile_navierstokes_P << q
   #G2D.vector().axpy(-1, nst.vector())
   #vtkfile_GENERAL_GRADE2 << project(G2D,V)
   #vtkfile_GENERAL_GRADE2 << project(q,Q)

#Enorm = norm(goldr.vector().axpy(-1, Uoldr.vector()),norm_type='H1')
#print(Enorm)

U.vector().axpy(-1,nst.vector())
vtkfile_navierstokes_dU << project(U,V)
