from mshr import *
from dolfin import *
import math,sys

from ufl import nabla_div
set_log_active(False)

pdeg = 3
fudg = 10000

reno = 1.
alfa = 0.01
#int(sys.argv[0])
#meshsize= 24 #int(sys.argv[1])
design = "classic" #str(sys.argv[2])
lbufr = -1; #float(sys.argv[3])
rbufr = 4; #float(sys.argv[4])
r0 = 0.5; #float(sys.argv[4])
r1 = 1; #float(sys.argv[4])
upright = 0.5
right = 0.5

mesh = Mesh("mesh.xml")

vtkfile_navierstokes_U = File('g2u.pvd')
vtkfile_navierstokes_P = File('g2p.pvd')

V = VectorFunctionSpace(mesh, "Lagrange", pdeg)
Q = FunctionSpace(mesh, "Lagrange", pdeg-1)

# define boundary condition
if False :
    boundary_exp = Expression(("exp(-fu*(lb-x[0])*(lb-x[0]))*(1.0-x[1]*x[1]) + \
      (1.0/up)*exp(-fu*(ri+rb-x[0])*(ri+rb-x[0]))*(1.0-((x[1]*x[1])/(up*up)))","0"), \
                       up=upright,ri=right,fu=fudg,rb=rbufr,lb=lbufr,degree = pdeg)
else :
    boundary_exp = Expression(("exp(-fu*(lb-x[0])*(lb-x[0]))*(1.0-(x[1]*x[1]+x[2]*x[2])) + \
      (1.0/up)*exp(-fu*(ri+rb-x[0])*(ri+rb-x[0]))*(1.0-((x[1]*x[1]+x[2]*x[2])/(up*up)))","0","0"), \
                       up=upright,ri=right,fu=fudg,rb=rbufr,lb=lbufr,degree = pdeg)
    

bc = DirichletBC(V, boundary_exp, "on_boundary")

# set the parameters
r = 1.0e4


""" GRADE TWO """

alpha_1 = Constant(alfa)
alpha_2 = Constant(-alfa)

Y = TensorFunctionSpace(mesh, "CG", 1, shape = (9,))
Ycg = TensorFunctionSpace(mesh, "CG", 1, shape = (3,3))
sigma = Function(Y)
#sigma = TestFunction(Y)
tau = TestFunction(Y)
tau_ = as_matrix(((tau[0], tau[1], tau[2]),
                  (tau[3], tau[4], tau[5]),
                  (tau[6], tau[7], tau[8])))


q = Function(Q)
u = TrialFunction(V)
v = TestFunction(V)

Uoldr = Function(V)
U = Function(V)

h = CellDiameter(mesh)
gg2_iter = -1
max_gg2_iter = 10
max_piter = 5

#inintial guess
U.vector()[:] = 1 # Should be NSE.
q.vector()[:] = 0



#Pre-define variables
incrnorm = 1; gtol = alfa*0.0001;  r = 1.0e4;
n = FacetNormal(mesh)

def A(z):
    return (grad(z) + grad(z).T)
        
sigma = Function(Y)
sigma.vector()[:] = 0


SIGMA_ = Function(Ycg)
SIGMA = Function(Y)
wgg2 = Function(V)
while gg2_iter < max_gg2_iter and incrnorm > gtol:
   wgg2.vector()[:] = 0

   piter = 0;  div_u_norm = 1
   gg2_iter += 1
   
   sigma_ = as_matrix( ((sigma[0], sigma[1], sigma[2]),
                        (sigma[3], sigma[4], sigma[5]),
                        (sigma[6], sigma[7], sigma[8])) )

   # FORMS FOR STRESS
   rz = inner(sigma_\
   + alpha_1*dot(U,    nabla_grad(sigma_)) \
   - (alpha_1*grad(U).T*A(U) \
   + (alpha_1 + alpha_2)*A(U)*A(U) \
   - reno*outer(U,U) \
   - alpha_1*q*grad(U).T \
   + alpha_1*SIGMA_*grad(U).T),tau_)*dx(mesh) \
   + 0.01*alpha_1*h*inner(nabla_grad(sigma_),nabla_grad(tau_))*dx(mesh) \
   #+ 0.01*alpha_1*h*inner(dot(U,nabla_grad(sigma_)), dot(U,nabla_grad(tau_)))*dx(mesh)
   #+ inner(0.5*alpha_1*div(U)*sigma_,tau_)*dx(mesh) \
   #+ abs(alpha_1*dot(U('-'),n('-')))*conditional(dot(U('-'),n('-'))<0,1,0)*inner(jump(sigma_),tau_('+'))*dS(mesh)

   print('solving for stress')
   solve(rz == 0, sigma, solver_parameters={"newton_solver": {"relative_tolerance": 1e-11}})# bcY,
   #sigma = Function(Y)
   #solve(T_gg2 == N_gg2, sigma)
   assign(SIGMA,sigma)
   #SIGMA_ = as_matrix(((SIGMA[0], SIGMA[1]), (SIGMA[2], SIGMA[3])))
   
   SIGMA_ = as_matrix( ((sigma[0], sigma[1], sigma[2]),
                       (sigma[3], sigma[4], sigma[5]),
                       (sigma[6], sigma[7], sigma[8])) )
   # FORMS FOR IPM
   a_gg2 = inner(grad(u), grad(v))*dx(mesh) + r*div(u)*div(v)*dx(mesh)
   #F_gg2 = inner(div(project(SIGMA_,Ycg)),v)*dx(mesh)
   F_gg2 = inner(div(SIGMA_),v)*dx(mesh) #inner(dot(SIGMA_,n),v)*ds-reno*inner(SIGMA_,grad(v))*dx(mesh) #
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
       incrnorm = errornorm(U,Uoldr,norm_type='H1',degree_rise=2)/norm(U,norm_type='H1')
       if(MPI.rank(mesh.mpi_comm()) == 0):
            print("div(u) ",piter , div_u_norm)
    
    
   
   # USA
   assign(q, project(-(div(wgg2)),Q))
   #q = project(-div(wgg2)/reno,Q) #/reno
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

   #vtkfile_GENERAL_GRADE2_sigma << project(sigma_,Ycg)
   #TAU = alfa*(A(U)*grad(U)+grad(U).T*A(U)+dot(U,nabla_grad(A(U))))-alfa*(A(U)*A(U))-reno*outer(U,U)
   #vtkfile_GENERAL_GRADE2_sigma << project(TAU,Ycg)
#G2P = Function(V)
#G2P.vector().axpy(1, SIGMA[0,0].vector())

#Enorm = norm(goldr.vector().axpy(-1, Uoldr.vector()),norm_type='H1')
#print(Enorm)
