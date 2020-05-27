
"""This program solves the grade-two non-Newtonian fluid model equations
in an expanding duct with Dirichlet boundary conditions given by
u=(1-r^2,0) on the left,  u = (1/0.5(1-(r/0.25)^2),0) on the right and
u=0 elsewhere.
The boundary conditions for z are posed only on the inflow boundary
"""
from mshr import *
from dolfin import *
import math,sys

from ufl import nabla_div
set_log_active(False)

meshsize=64  #int(sys.argv[1])
reno=2. #float(sys.argv[2])
alfa=.1 #float(sys.argv[3])
#rbufr=float(sys.argv[4])

domain_bool = "none" #(sys.argv[4])

mesh = Mesh("mesh.xml")


def in_bdry(x):
  return x[0] <= lbufr+DOLFIN_EPS
  
def out_bdry(x):
  return x[0] > rbufr+right-DOLFIN_EPS
  
def calculate_pressure_drop(p_in, V, mesh):
# Outline for pressure drop calculations
  p = project(p_in, FunctionSpace(mesh, "Lagrange", pdeg - 1))

  # Analytical velocity profile and approximated "marker function"
  velocity_in = interpolate(Expression(("(0.5*(1+tanh(K*(-x[0]))))*-x[0]/(2*bi)","0"), K = 1000, bi = lbufr, degree=4), V)
  velocity_out = interpolate(Expression(("(0.5*(1+tanh(K*(x[0]-L))))*(x[0]-L)/(2*H*b0)","0"), K = 1000, L = 1.0, H = 0.5, b0 = rbufr, degree=4), V)

  p_avg_in_form = div(velocity_in*p)*dx(mesh)
  p_avg_in = assemble(p_avg_in_form)

  p_avg_out_form = div(velocity_out*p)*dx(mesh)
  p_avg_out = assemble(p_avg_out_form)

  #print('p_delta: ', p_avg_in-p_avg_out,'\n')
  return p_avg_in-p_avg_out
  
def sigma_f(U,P):
    return 2*sym(grad(U)) - P*Identity(2)
  
def calculate_total_force(u, p_in, mesh):
    p = project(p_in, FunctionSpace(mesh, "Lagrange", pdeg - 1))
    ZZ = FunctionSpace(mesh, "Lagrange", pdeg)
    n = FacetNormal(mesh)
    v = interpolate(Expression(("1.","0."),degree = pdeg),V)
    # approximated heaviside marker for the nozzle in the axial direction, this should not be needed
    #marker_nozzle = interpolate(Expression(("0.5*(1+tanh(K*(-x[0]))-(1+tanh(K*(x[0]-H))))"), K = 10000, H = H, lb = lbufr, degree=4), Z)
    marker_nozzle = interpolate(Expression(("((x[0]>0)&&(x[0]<H))"),H = H, degree=0), Z)
    Force = marker_nozzle*dot(-n,2*sym(grad(u))*v)*ds(mesh)
    PressureForce = marker_nozzle*p*dot(n,v)*ds(mesh)
    
    TotalForce = marker_nozzle*dot(dot(sigma_f(u,p),n),-v)*ds(mesh)
    return assemble(Force), assemble(PressureForce), assemble(TotalForce)

vtkfile_stokes = File('ust.pvd')
vtkfile_navierstokes = File('unst.pvd')
vtkfile_grade2 = File('ug2.pvd')

vtkfile_dnavierstokes = File('dunst.pvd')
vtkfile_dstokes = File('dstokes.pvd')
vtkfile_navierstokes_p = File('pnst.pvd')

vtkfile_GENERAL_GRADE2 = File('dug2general.pvd')
vtkfile_GENERAL_GRADE2_sigma = File('sigmag2general.pvd')


V = VectorFunctionSpace(mesh, "Lagrange", pdeg)
Z = FunctionSpace(mesh, "Lagrange", pdeg)
# define boundary condition

if fales :
    boundary_exp = Expression(("exp(-fu*(lb-x[0])*(lb-x[0]))*(1.0-x[1]*x[1]) + \
      (1.0/up)*exp(-fu*(ri+rb-x[0])*(ri+rb-x[0]))*(1.0-((x[1]*x[1])/(up*up)))","0"), \
                       up=upright,ri=right,fu=fudg,rb=rbufr,lb=lbufr,degree = pdeg)
else
    boundary_exp = Expression(("exp(-fu*(lb-x[0])*(lb-x[0]))*(1.0-x[1]*x[1]+x[2]*x[2]) + \
      (1.0/up)*exp(-fu*(ri+rb-x[0])*(ri+rb-x[0]))*(1.0-((x[1]*x[1]+x[2]*x[2])/(up*up)))","0"), \
                       up=upright,ri=right,fu=fudg,rb=rbufr,lb=lbufr,degree = pdeg)


bc = DirichletBC(V, boundary_exp, "on_boundary")
#bczee = DirichletBC(Z, boundary_zee, "on_boundary")
bczee = DirichletBC(Z, boundary_zee, in_bdry)
# set the parameters
zf = Expression(("0","0"), degree = pdeg)
bcz = DirichletBC(V, zf, "on_boundary")
r = 1.0e4
# define test and trial functions, and function that is updated
uold = TrialFunction(V)
gold = TrialFunction(V)
zee = TrialFunction(Z)
v = TestFunction(V)
#w = Function(V)
#w = project(zf,V)
asf = inner(grad(uold),grad(v))*dx + r*div(uold)*div(v)*dx
bs = -div(w)*div(v)*dx
uold = Function(V)
ust = Function(V)
pdes = LinearVariationalProblem(asf, bs, uold, bc)
solvers = LinearVariationalSolver(pdes)
# Stokes solution
# Scott-Vogelius iterated penalty method
iters = 0; max_iters = 5; div_u_norm = 1
while iters < max_iters and div_u_norm > 1e-10:
# solve and update w
    solvers.solve()
    w.vector().axpy(r, uold.vector())
# find the L^2 norm of div(u) to check stopping condition
    div_u_norm = sqrt(assemble(div(uold)*div(uold)*dx(mesh)))
    print( "   IPM iter_no=",iters,"div_u_norm="," %.5e"%div_u_norm)
    iters += 1
print( "Stokes solver  IPM iter_no=",iters,"div_u_norm="," %.2e"%div_u_norm)
pstoke = calculate_pressure_drop(-div(w)/reno, V, mesh)
fstoke, fpstoke, ftstoke = calculate_total_force(uold,-1/reno*div(w), mesh)

uold.set_allow_extrapolation(True)
seeu=interpolate(uold,W)
#plot(seeu[0], interactive=True)

ust.vector().axpy(1.0, uold.vector())
vtkfile_stokes << ust
# ust is the Stokes solution
# Navier-Stokes solver
kters = 0; max_kters = 9; unorm = 1
while kters < max_kters and unorm > 1e-6:
    u = TrialFunction(V)
    v = TestFunction(V)
    w = Function(V)
    w = project(zf,V)
#   uold = Function(V)
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
        print( "   IPM iter_no=",iters,"div_u_norm="," %.2e"%div_u_norm)
        iters += 1
    print( "   IPM iter_no=",iters,"div_u_norm="," %.2e"%div_u_norm)
    kters += 1
    uold.vector().axpy(-1.0, u.vector())
#   plot(uold[0], interactive=True)
    unorm=norm(u,norm_type='H1')
    uoldnorm=norm(uold,norm_type='H1')
    div_u_norm = sqrt(assemble(div(uold)*div(uold)*dx(mesh)))
    print( "Newton iter_no=",kters,"Delta_u_norm="," %.2e"%unorm,"u_norm=", \
          " %.2e"%uoldnorm,"div_u_norm="," %.2e"%div_u_norm)

# Stoke's projection
uz = TrialFunction(V)
v = TestFunction(V)
w = Function(V)
w = project(zf,V)

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
pnse = calculate_pressure_drop(-1/reno*div(w), V, mesh)
fnse, fpnse, ftnse = calculate_total_force(uz,-1/reno*div(w),mesh)
vtkfile_navierstokes << uz

#plot(uold[0], interactive=True)
nst = Function(V)
# nst is the Navier-Stokes solution
#nst.vector().axpy(1.0, uold.vector())
nst.vector().axpy(1.0, uz.vector())
gold = Function(V)
gold.vector().axpy(1.0, uz.vector())
#gold.vector().axpy(1.0, uz.vector())
# Grade-two solver
gters = 0; max_gters = 20; incrgoldnorm = 1
errfn = Function(V)
goldr = Function(V)
gtol=alfa*0.0001

print( "z_BC's upr","lbuf","rbuf","MS","Re_no"," alpha ","  NSt-Gtu","  Sto-Gtu","  NSt-Sto"," gits", \
      "G2_it_err","|Stokes|","|z|_L2")
print( "inflow ",upright,lbufr,rbufr,meshsize,"%.1f"%reno," %.2e"%alfa," %.2e"%goldnstnorm, \
       " %.2e"%stomingtoonorm, " %.2e"%stominavstorm,"  ",gters,"   %.1e"%incrgoldnorm, \
       " %.4f"%goldnorm," %.4f"%norm(zee,norm_type='L2'))
#plot(zee, interactive=True)
plot(gold[0], interactive=True)
vtkfile_dnavierstokes << gold

""" GRADE TWO """

alpha_1 = Constant(alfa)
alpha_2 = Constant(-alfa)

Y = TensorFunctionSpace(mesh, "CG", pdeg, shape = (9,))
Ycg = TensorFunctionSpace(mesh, "CG", pdeg, shape = (2,2))
A = grad(goldr) + grad(goldr).T
TAU = alfa*(A*grad(goldr)+grad(goldr).T*A+dot(goldr,nabla_grad(A)))-alfa*(A*A)-reno*outer(goldr,goldr)
vtkfile_dnavierstokes << project(TAU,Ycg)

#Y = TensorFunctionSpace(mesh, "Lagrange", pdeg)
Q = FunctionSpace(mesh, "Lagrange", pdeg-1)
ZZ = FunctionSpace(mesh, "DG", 0)
zz = TestFunction(ZZ)

ZeroV = Expression(("0","0"),degree = pdeg)
ZeroQ = Expression(("0"), degree = pdeg)
ZeroY = Expression((("0","0"),("0","0")), degree = pdeg)


StressINflow = Expression((("exp(-fu*(lb-x[0])*(lb-x[0]))*(4*x[1]*x[1]*a2-(1.0-x[1]*x[1])*(1.0-x[1]*x[1]))",\
                            "0",\
                            "0",\
                            "exp(-fu*(lb-x[0])*(lb-x[0]))*(4*x[1]*x[1]*(2*a1+a2))")),up = upright, fu = fudg, lb = lbufr, a1 = alfa, a2 = -alfa, re = reno, degree = pdeg)

StressOUTflow = Expression((("exp(-fu*(lb-x[0])*(lb-x[0]))*(4*x[1]*x[1]*a2/(up*up*up*up)-(1.0-x[1]*x[1]/(up*up))*(1.0-x[1]*x[1]/(up*up)))",\
                            "0",\
                            "0",\
                            "exp(-fu*(lb-x[0])*(lb-x[0]))*(4*x[1]*x[1]/(up*up*up*up)*(2*a1+a2))")),up = upright, fu = fudg, lb = lbufr, a1 = alfa, a2 = -alfa, re = reno, degree = pdeg)


#bcYi = DirichletBC(Y, BCFU,  in_bdry, 'pointwise') #"on_boundary")#
#bcYo = DirichletBC(Y, StressOUTflow, out_bdry, 'pointwise')

#bcY = [bcYi, bcYo]

sigma = Function(Y)
#sigma = TestFunction(Y)
tau = TestFunction(Y)
tau_ = as_matrix(((tau[0], tau[1]),
                  (tau[2], tau[3])))

Xn = as_vector((1,0))

q = Function(Q)

u = TrialFunction(V)
v = TestFunction(V)

Uoldr = Function(V)
U = Function(V)

h = CellDiameter(mesh)
gg2_iter = -1
max_gg2_iter = 10
max_piter = 10

#inintial guess
assign(U, project(nst,V))
assign(q, project(ZeroQ,Q))
SIGMA_ = project(ZeroY,Ycg)

#Pre-define variables
incrnorm = 1; gtol = gtol;  r = 1.0e4;
n = FacetNormal(mesh)

# Expression for cell inflow


def A(z):
    return (grad(z) + grad(z).T)
    
    
sigma = Function(Y)
sigma.vector()[:] = 0

SIGMA = Function(Y)
wgg2 = Function(V)
while gg2_iter < max_gg2_iter and incrnorm > gtol:
   wgg2.vector()[:] = 0

   piter = 0;  div_u_norm = 1
   gg2_iter += 1
   
   sigma_ = as_matrix(((sigma[0], sigma[1]), \
                       (sigma[2], sigma[3])))


   # FORMS FOR STRESS
   rz = inner(sigma_\
   + alpha_1*dot(U,    nabla_grad(sigma_)) \
   - (alpha_1*grad(U).T*A(U) \
   + (alpha_1 + alpha_2)*A(U)*A(U) \
   - reno*outer(U,U) \
   - alpha_1*q*grad(U).T \
   + alpha_1*SIGMA_*grad(U).T),tau_)*dx(mesh) \
   + 0.01*alpha_1*h*inner(nabla_grad(sigma_),nabla_grad(tau_))*dx(mesh) \
   + 0.01*alpha_1*h*inner(dot(U,nabla_grad(sigma_)), dot(U,nabla_grad(tau_)))*dx(mesh)
   #+ inner(0.5*alpha_1*div(U)*sigma_,tau_)*dx(mesh) \
   #+ abs(alpha_1*dot(U('-'),n('-')))*conditional(dot(U('-'),n('-'))<0,1,0)*inner(jump(sigma_),tau_('+'))*dS(mesh)

   print('solving for stress')
   solve(rz == 0, sigma, solver_parameters={"newton_solver": {"relative_tolerance": 1e-11}})# bcY,
   #sigma = Function(Y)
   #solve(T_gg2 == N_gg2, sigma)
   assign(SIGMA,sigma)
   SIGMA_ = as_matrix(((SIGMA[0], SIGMA[1]), (SIGMA[2], SIGMA[3])))
                       
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
   
       print("div(u) ",piter , div_u_norm)
    
    
   
   # USA
   assign(q, project(-(div(wgg2)),Q))
   #q = project(-div(wgg2)/reno,Q) #/reno
   
   print( "   IPM iter_no=",piter,"div_u_norm="," %.2e"%div_u_norm)
   print( gg2_iter,"change="," %.3e"%incrnorm)
   #print(errornorm(SIGMA,sigma0,norm_type='H1',degree_rise=2)/norm(SIGMA,norm_type='H1'))
   


   
   Uoldr = Function(V)
   Uoldr.vector().axpy(1, U.vector())

   G2D = Function(V)
   G2D.vector().axpy(1, U.vector())
    #assign(G2D,U)

   G2D.vector().axpy(-1, nst.vector())
   vtkfile_GENERAL_GRADE2 << project(G2D,V)
   #vtkfile_GENERAL_GRADE2 << project(q,Q)

   vtkfile_GENERAL_GRADE2_sigma << project(sigma_,Ycg)
   TAU = alfa*(A(U)*grad(U)+grad(U).T*A(U)+dot(U,nabla_grad(A(U))))-alfa*(A(U)*A(U))-reno*outer(U,U)
   vtkfile_GENERAL_GRADE2_sigma << project(TAU,Ycg)
#G2P = Function(V)
#G2P.vector().axpy(1, SIGMA[0,0].vector())

#Enorm = norm(goldr.vector().axpy(-1, Uoldr.vector()),norm_type='H1')
#print(Enorm)


print('to tex')
print("%.1f"%reno,"& %.2f"%alfa,"& %.2e"%goldnstnorm, \
"& %.2e"%stomingtoonorm, "& %.2e"%stominavstorm, \
"& %.4f"%norm(zee,norm_type='L2'),"& %.3f"%pstoke,"& %.3f"%pnse,"& %.3f"%pg, "\\\\")

print("%.1f"%reno,"& %.2f"%alfa,"& %.2e"%goldnstnorm, \
"& %.2e"%stomingtoonorm, "& %.2e"%stominavstorm, \
"& %.4f"%norm(zee,norm_type='L2'),"& %.3f"%fstoke,"& %.3f"%fnse,"& %.3f"%fg, "\\\\")

print("%.1f"%reno,"& %.2f"%alfa,"& %.2e"%goldnstnorm, \
"& %.2e"%stomingtoonorm, "& %.2e"%stominavstorm, \
"& %.4f"%norm(zee,norm_type='L2'),"& %.3f"%fpstoke,"& %.3f"%fpnse,"& %.3f"%fpg, "\\\\")

print("%.1f"%reno,"& %.2f"%alfa,"& %.2e"%goldnstnorm, \
"& %.2e"%stomingtoonorm, "& %.2e"%stominavstorm, \
"& %.4f"%norm(zee,norm_type='L2'),"& %.3f"%ftstoke,"& %.3f"%ftnse,"& %.3f"%ftg, "\\\\")
