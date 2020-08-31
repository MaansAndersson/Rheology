
from mshr import *
from dolfin import *
import math,sys

eps = 1e-6
lbufr = -1; #float(sys.argv[3])
rbufr = 3; #float(sys.argv[4])
r0 = 0.5; #float(sys.argv[4])
r1 = 1; #float(sys.argv[4])
upright = .5 #0.5
right = 1.0 #0.5

stab = "GLS" #"SLD"
TimeDep = True
Oldroyd = False #False
Grade2 = True #True

Rheomodel = Oldroyd or Grade2

if Oldroyd:
    print('Oldroyd-B solver')

if Grade2:
    print('Grade two solver')
else:
    print('NSE solver')

if (True):
    vtkfile_u = File('non-newtonian_files/u.pvd')
    vtkfile_p = File('non-newtonian_files/p.pvd')
    
# Stenosed pipe geometry.

mesh = Mesh("mesh.xml")
dim = mesh.geometric_dimension()

pdeg = 1
V = VectorFunctionSpace(mesh, "CG", pdeg + 1)
Q = FunctionSpace(mesh, "CG", pdeg)
Y = TensorFunctionSpace(mesh, "CG", pdeg)
Z = FunctionSpace(mesh, "DG", 0)
    
# SubDomains
class inflow(SubDomain):
    def inside(self, x, on_boundary):
        return x[0]<lbufr+eps and on_boundary
class outflow(SubDomain):
    def inside(self, x, on_boundary):
        return x[0]>right+rbufr-eps and on_boundary
class noslip_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return  (x[0] <= right+rbufr-eps) and (x[0]>lbufr+eps) and on_boundary



# Boundary Conditions
INFLOW_EXPRESSION = Expression(("(1-x[1]*x[1])","0"), degree = pdeg+1)
bcm1 = DirichletBC(V, INFLOW_EXPRESSION, inflow())
bcm2 = DirichletBC(V, Expression(("0","0"), degree = pdeg+1), noslip_boundary())
bcm = [bcm1, bcm2]
bcc = DirichletBC(Q, Expression(("0"), degree = pdeg+1), outflow())

# Expression for Initial Guess
ZeroV = Expression(("0","0"),degree = 1)
ZeroQ = Expression(("0"), degree = 1)
ZeroY = Expression((("0","0"),("0","0")), degree = 1)

# Define Functions
v = TestFunction(V)
q = TestFunction(Q)
u = Function(V)
p = Function(Q)
u0 = Function(V)
p0 = Function(Q)
umax = Constant(1)
U = Function(V)
P = Function(Q)
u_ = TrialFunction(V)
p_ = TrialFunction(Q)

# Initiate functions
u = project(ZeroV,V)
u0 = project(ZeroV,V)
v0 = project(ZeroV,V)
p = project(ZeroQ,Q)
p0 = project(ZeroQ,Q)


""" Oldroyd Stresses - temp """
y = TestFunction(Y)
T0 = Function(Y)
Tp = Function(Y)
TT = Function(Y)
T_ = TrialFunction(Y)

Tp = project(ZeroY,Y)
T0 = project(ZeroY,Y)

""" Parameters """
hmin = mesh.hmin()
k = hmin*0.1
nu = 1/1
h = CellDiameter(mesh)

t = 0 #initial time
if TimeDep:
    T = 50*(hmin*0.1)/2 ; #Final time
else:
    T = (hmin*0.1)/2 #Step into the loop
    
    
beta = 1.0
if Oldroyd:
    lambda_1 = 0.1
    beta = 0.99
elif Grade2:
    alpha_1 = 0.1
    alpha_2 = -0.1
    

MAXIT = 15; # Max iterations for fixed point iteration

d1 = 2*hmin*h #1e-3 #0.01
d2 = 2*hmin*h
c1 = 2*h*hmin
c2 = 2*h*hmin

""" Simulation """

while ( t < T ):

  if(TimeDep):
    um = 0.5*(u + u0)
  else:
    um = u
    
    """ Stationary NS Equations """
  rm = (beta*nu*inner(grad(um), grad(v)) + inner(grad(p) + grad(um)*um, v))*dx
  rc = (inner(div(um), q))*dx
  


  """ Oldroyd-B Stresses - temp """
  if ( Oldroyd ):
    rt = inner(Tp,y)*dx(mesh) \
                    + lambda_1*(inner(dot(um,grad(Tp)),y) \
                    - inner(grad(um)*Tp,y) \
                    - inner(T*grad(um).T,y))*dx(mesh) \
                    - inner(2*nu*(1-beta)*sym(grad(um)),y)*dx(mesh)
  
  """ Grade2 Stress - temp """
  if ( Grade2 ):
    A = 2*sym(grad(um))
    rt = inner(Tp,y)*dx(mesh) -\
         inner(alpha_1*(dot(um, grad(A)) + A*grad(um) + grad(um).T*A)
         + alpha_2*A*A, y)*dx(mesh)
    
    
  if ( TimeDep ):
    rm += inner(u - u0, v)/k*dx - inner(div(Tp - T0),v)*dx
    rc += (k*inner(grad(p - p0), grad(q)))*dx
    if (Rheomodel):
        rt += inner((Tp - T0),y)/k*dx(mesh)
  else:
    if ( Rheomodel ):
      rm += -inner(div(Tp),v)*dx
    rc += 1/nu*inner(grad(p),grad(q))*dx
  
  r1 = rm
  r2 = rc
  if (Rheomodel):
    r3 = rt
  # NSE Stabilisation schemes
  if (stab == "GLS"): # Galerkin Least Square stabilisation
    rm += d1*(inner(grad(p) + grad(um)*um, grad(v)*um) + inner(div(um), div(v)))*dx
    rc += d2*inner(grad(p) + grad(um)*um, grad(q))*dx #+ d2*inner(p, q)*dx
  if (stab == "SLD"): #Streamline Diffusion stabilisation
    rm += c1*inner(grad(um)*um , grad(v)*um)*dx
    rc += c2*inner(grad(p),grad(q))*dx #+ c2*inner(p, q)*dx
    
  """ Prepare Newton-Step """
  Jm = derivative(rm, u, u_)
  Jc = derivative(rc, p, p_)


  am = Jm
  Lm = action(Jm, u) - rm

  ac = Jc
  Lc = action(Jc, p) - rc


  problemM = LinearVariationalProblem(am, Lm, U, bcm)
  solverm = LinearVariationalSolver(problemM)

  problemC = LinearVariationalProblem(ac, Lc, P, bcc)
  solverc = LinearVariationalSolver(problemC)

  """ Non Newtonian Stresses - temp """
  if (Rheomodel):
      Jt = derivative(rt, Tp, T_)
      at = Jt
      Lt = action(Jt, Tp) - rt
      problemT = LinearVariationalProblem(at, Lt, TT)
      solvert = LinearVariationalSolver(problemT)
    
  # Newton Step
  for i in range(0, MAXIT):
    residual_m = assemble(r1)
    residual_c = assemble(r2)
    print('Residual momentum: ', residual_m.norm('l2'))
    print('Residual continuity: ', residual_c.norm('l2'))
    if(Rheomodel):
        residual_t = assemble(r3)
        print('Residual stress: ', residual_t.norm('l2'))

    solverm.solve()
    assign(u, U)
    solverc.solve()
    assign(p, P)
    ## Check Convergance
    if not TimeDep:
        vtkfile_u << u
        vtkfile_p << p


    if (Rheomodel) and i > -1:
      solvert.solve()
      assign(Tp, TT)
        
    #if (rlative_change < 1e-6)
    #    In the different c-equations
    
    
    if TimeDep:
      assign(u0, u)
      assign(p0, p)
      if Rheomodel:
        assign(T0, Tp)
  t += k
  print(t)
  if TimeDep:
    vtkfile_u << u
    vtkfile_p << p
