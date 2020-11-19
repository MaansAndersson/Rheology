from mshr import *
from dolfin import *
import math,sys
from scipy.io import loadmat, savemat
from ufl import nabla_div
from timeit import default_timer as timer
#set_log_active(False)

pdeg = 4
fudg = 10000*1000
reno = float(sys.argv[1])
alfa = float(sys.argv[2]) #0.01
lbufr = -1; #float(sys.argv[3])
rbufr = 3; #float(sys.argv[4])
r0 = 0.5; #float(sys.argv[4])
r1 = 1; #float(sys.argv[4])
right = 1.0 #0.5

q_base = 0
pipe = str(sys.argv[3])
if pipe == 'pipe':
    pipe = True
else:
    pipe = False
    
if pipe:
    upright = 1.
    right = 1 #0.5 #0.5
else:
    upright = 0.5
    right = 1
    
L = (-lbufr + right + rbufr)


alpha_1 = Constant(alfa)
alpha_2 = Constant(-alfa)

mesh = Mesh("mesh.xml") 
dim = mesh.geometric_dimension()
print(dim)
cell = mesh.ufl_cell()
print("pdeg: ", pdeg)

Vp = VectorFunctionSpace(mesh, "Lagrange", pdeg)
# Refine inlet and outlet
for i in range(0,0):
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    for cell in cells(mesh):
      cell_markers[cell] = False
      p = cell.midpoint()
      if (abs(p.y()-0) > (0.5)-0.04 and (p.x()-1)>0) or abs(p.y()-0) > 1-0.04:
          cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)

for i in range(0,0): #3): #5):
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    radius  = 0.5/(i+1) #-0.01*(i)
    for cell in cells(mesh):
      cell_markers[cell] = False
      p = cell.midpoint()
      if (p.x()-1.0)**2 + (p.y()+0.65)**2 < radius**2 or (p.x()-1.0)**2 + (p.y()-0.65)**2 < radius**2:
          cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)
    
for i in range(0,0): #5):
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    radius  = 0.2 #-0.01*(i)
    for cell in cells(mesh):
      cell_markers[cell] = False
      p = cell.midpoint()
      if (p.x()-0.0)**2 + (p.y()+1.05)**2 < radius**2 or (p.x()-0)**2 + (p.y()-1.05)**2 < radius**2:
          cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)
    
#mesh = refine(mesh)
    
print(len(mesh.cells()))
vtkfile_M = File('results/mesh.pvd')
vtkfile_M << mesh
vtkfile_U = File('results/g2u.pvd')
vtkfile_P = File('results/g2p.pvd')
vtkfile_W = File('results/g2W.pvd')
vtkfile_dU = File('results/grade2W-nse.pvd')

Bubble = False
if Bubble:
    # some demands on pdeg.
    pdeg = 2
    V1 = FiniteElement("CG", mesh.ufl_cell(), pdeg)
    B = FiniteElement("B", mesh.ufl_cell(), mesh.topology().dim() + 1)
    V = FunctionSpace(mesh, VectorElement(NodalEnrichedElement(V1, B)))
    V2 = VectorFunctionSpace(mesh, "CG", pdeg) #FunctionSpace(mesh, VectorElement(NodalEnrichedElement(V1, B))) #
    Q = FunctionSpace(mesh, "Lagrange", pdeg)
    Q1 = FunctionSpace(mesh, "Lagrange", pdeg)
else:
    V = VectorFunctionSpace(mesh, "Lagrange", 2+pdeg)
    Q = FunctionSpace(mesh, "Lagrange", pdeg+2-1)
    Q1 = FunctionSpace(mesh, "Lagrange", pdeg)
    #V1 = FiniteElement("CG", mesh.ufl_cell(), pdeg)
    V2 = VectorFunctionSpace(mesh, "CG", pdeg-3)
    #B = FiniteElement("B", mesh.ufl_cell(), mesh.topology().dim() + 1)
    #V2 = FunctionSpace(mesh, VectorElement(NodalEnrichedElement(V1, B)))


def in_bdry(x):
    return x[0] <= lbufr+DOLFIN_EPS
def out_bdry(x):
    return x[0] >= right + rbufr-DOLFIN_EPS

# define boundary condition
if dim == 2 :
    boundary_exp = Expression(("exp(-fu*(lb-x[0])*(lb-x[0]))*(1.0-x[1]*x[1]) + \
      (1.0/up)*exp(-fu*(ri+rb-x[0])*(ri+rb-x[0]))*(1.0-((x[1]*x[1])/(up*up)))","0"), \
                       up=upright,ri=right,fu=fudg,rb=rbufr,lb=lbufr,degree = pdeg)
                       
    if pipe:
        WINFLOW = Expression(("0","r*4*x[1]*(3*a1+2*a2)"), a1 = alpha_1, a2 = alpha_2, r = reno, degree = pdeg)
        WOUTFLOW = Expression(("0","r*4*(x[1])*(3*a1+2*a2)"), a1 = alpha_1, a2 = alpha_2, r = reno, degree = pdeg)
    else:
        WINFLOW = Expression(("0","r*4*x[1]*(3*a1+2*a2)"), a1 = alpha_1, a2 = alpha_2, r = reno, degree = pdeg)
        WOUTFLOW = Expression(("0","r*4*(x[1]/(up*up))*(3*a1+2*a2)"), a1 = alpha_1, a2 = alpha_2, r = reno, up = upright, degree = pdeg)
        
    UINFLOW = Expression(("(1.0-x[1]*x[1])","0"),up=upright,ri=right,fu=fudg,rb=rbufr,lb=lbufr,degree = pdeg)
    Analytic_q = Expression(( "-2*(x[0]) + r*((2*a1+a2)*(4*x[1]*x[1]) + 2*a1*(1-x[1]*x[1])) - q_base" ), degree=pdeg+1, a1=alpha_1, a2=alpha_2, lb = lbufr, rb = rbufr, r = reno, U = 1, L = L, q_base = q_base)
    

else :
    boundary_exp = Expression(("exp(-fu*(lb-x[0])*(lb-x[0]))*(1.0-(x[1]*x[1]+x[2]*x[2])) + \
      (1.0/up)*exp(-fu*(ri+rb-x[0])*(ri+rb-x[0]))*(1.0-((x[1]*x[1]+x[2]*x[2])/(up*up)))","0","0"), \
                       up=upright,ri=right,fu=fudg,rb=rbufr,lb=lbufr,degree = pdeg)
    

bc = DirichletBC(V, boundary_exp, "on_boundary")
bcW1 = DirichletBC(V2, WINFLOW,  in_bdry, 'pointwise')
bcW2 =  DirichletBC(V2, WINFLOW,  out_bdry) #, 'pointwise')

bcW = [] # [bcW1] #[bcW1, bcW2]
# set the parameters
r = Constant(0*1/20)  # time-step artificial compressability
rp = Constant(0*2.0) # step-length artificial compressability
ro = Constant(1.e4)   # penelty in penelty iteration

W = Function(V2)
Wtemp = Function(V2)
W_ = TrialFunction(V2)
q = Function(Q)
u = TrialFunction(V)
v = TestFunction(V)
v2 = TestFunction(V2)
uold = Function(V)
Uoldr = Function(V)
U = project(Function(Vp,'nst.xml'),V)
P = Function(Q)
nst = Function(V)
nst.vector()[:] = U.vector()[:]

if False:
    uvec_old = loadmat('nst')['uvec']
    nst.vector().set_local(uvec_old[:,0])
    U.vector().axpy(1, nst.vector())
#inintial guess
q.vector()[:] = 0

infl = Expression("x[0] <= lb", lb = lbufr, degree = pdeg)

h = CellDiameter(mesh)
hf = FacetArea(mesh)
gg2_iter = -1
max_gg2_iter = 5
max_piter = 15


#Pre-define variables
incrnorm = 1; gtol = 5e-5 #alfa*0.0000001; #alfa*0.0001;  r = 1.0e4;
n = FacetNormal(mesh)

def A(z):
    return (grad(z) + grad(z).T)

wgg2 = Function(V)


while gg2_iter < max_gg2_iter and incrnorm > gtol:
   wgg2.vector()[:] = 0

   piter = 0;  div_u_norm = 1; gg2_iter += 1
   
   # FORMS FOR STRESS
   
   rz =  inner(1/reno*W_ \
     + Constant(1.0)*alpha_1*dot(U, nabla_grad(W_)) \
     - div(alpha_1*grad(U).T*A(U) \
     + (alpha_1 + alpha_2)*A(U)*A(U) \
     - outer(U,U) \
     - alpha_1*grad(U).T*q), v2)*dx(mesh) \
     + Constant(0.0)*alpha_1*h*inner(grad(W_), grad(v2))*dx(mesh) \
     + Constant(1)*alpha_1*h*inner(dot(U,nabla_grad(W_)), dot(U,nabla_grad(v2)))*dx(mesh) \
     - Constant(0)*alpha_1*inner(dot(U, nabla_grad(v2)),W_)*dx \
     + alpha_1*0.5*inner(div(U)*W,v2)*dx
# + infl*Constant(0)/h*inner(W_-WINFLOW, v2)*ds(mesh) 
#+ Constant(0)*abs(alpha_1*dot(U('-'),n('-')))*conditional(dot(U('-'),n('-'))<0,1,0)*inner(jump(W_),v2('+'))*dS(mesh) \



   rz += Constant(0)*h*inner(W_ + alpha_1*dot(U, nabla_grad(W_)) \
                                    - div(alpha_1*grad(U).T*A(U)   \
                                    + (alpha_1 + alpha_2)*A(U)*A(U)\
                                    - reno*outer(U,U) \
                                    - alpha_1*q*grad(U).T), \
                                v2 + alpha_1*dot(U, nabla_grad(v2)) \
                                    - div(alpha_1*grad(U).T*A(U) \
                                    + (alpha_1 + alpha_2)*A(U)*A(U) \
                                    - reno*outer(U,U) \
                                    - alpha_1*q*grad(U).T))*dx
   
   """
   rz =  inner(1/reno*W_ \
     + alpha_1*dot(U, nabla_grad(W_))\
     - div(alpha_1*grad(U).T*A(U) \
     + (alpha_1 + alpha_2)*A(U)*A(U)), v2)*dx(mesh) \
     + inner(dot(U,nabla_grad(U)),v2)*dx(mesh) \
     + inner(alpha_1*grad(U).T*grad(q),v2)*dx(mesh) \
     + Constant(0.0)*alpha_1*h*inner(grad(W_), grad(v2))*dx(mesh) \
   #rz += Constant(10*0.1)*alpha_1*h*inner(dot(U,nabla_grad(W_)), dot(U,nabla_grad(v2)))*dx(mesh)
   #rz += Constant(10*0.1)*alpha_1*h*inner(dot(W_,nabla_grad(U)), dot(v2,nabla_grad(U)))*dx(mesh)
   """
   
   aa = lhs(rz)
   bb = rhs(rz)
   Aa, Bb = assemble_system(aa, bb, bcW)
   

   print('solving for stress')
   
   solve(Aa, W.vector(), Bb, 'lu')
   
   # If newtoniteartion
   #for ijk in range(2):
   #    J = derivative(rz, W_)
   #    Aaa, Bbb = assemble_system(J, -rz, bcW)
   #    solve(Aaa, Wtemp.vector(), Bbb, 'lu')
   #    W.vector()[:] += Wtemp.vector()[:]


#   solve(rz == -1, W, bcW, solver_parameters={"newton_solver": {"relative_tolerance": 1e-12}}) #bcW

   # FORMS FOR IPM
   a_gg2 = inner(grad(u),grad(v))*dx + r*inner(u,v)*dx + ro*div(u)*div(v)*dx
   F_gg2 = inner(W,v)*dx(mesh)
   b_gg2 = r*inner(U,v)*dx - div(wgg2)*div(v)*dx  #inner(grad(div(w)),v)*dx +
   

   """
   a_gg2 = inner(grad(u), grad(v))*dx(mesh) + r*div(u)*div(v)*dx(mesh)
   F_gg2 = inner(W,v)*dx(mesh)
   b_gg2 = -div(wgg2)*div(v)*dx(mesh)
   """
   
   # Solver IPM
   #pdegg2 = LinearVariationalProblem(a_gg2, F_gg2 + b_gg2, U, bc)
   #solvergg2 = LinearVariationalSolver(pdegg2)


   while (piter < max_piter and div_u_norm > 1.0E-10):
       #solvergg2.solve()
       start = timer()
       Am, bm = assemble_system(a_gg2, F_gg2 + b_gg2, bc)
       end = timer()
       #if(MPI.rank(mesh.mpi_comm()) == 0):
       #    print('      Assemble time: ', end - start)
           
       start = timer()
       solve(Am, U.vector(), bm, 'lu')
       end = timer()
       #if(MPI.rank(mesh.mpi_comm()) == 0):
       #    print('      Linear solver time: ', end - start)
           
       wgg2.vector().axpy(rp+ro, U.vector())
       
       # Criteria for stopping
       div_u_norm = sqrt(assemble(div(U)*div(U)*dx(mesh)))
       piter += 1
       incrnorm = errornorm(U,Uoldr,norm_type='H1',degree_rise=0)/norm(U,norm_type='H1')
       if(MPI.rank(mesh.mpi_comm()) == 0):
            print("div(u) ",piter , div_u_norm)
    
   assign(q, project(-(div(wgg2)),Q))
   if(MPI.rank(mesh.mpi_comm()) == 0):
       print( "   IPM iter_no=",piter,"div_u_norm="," %.2e"%div_u_norm)
       print( gg2_iter,"change="," %.3e"%incrnorm)


   Uoldr = Function(V)
   Uoldr.vector().axpy(1, U.vector())

   G2D = Function(V)
   G2D.vector().axpy(1, U.vector())
    #assign(G2D,U)

   vtkfile_U << project(U,V)

    
# For a 2D pipe
#Analytic_pressure = Expression(( "-2*((x[0]-1.5)) + (2*a1+a2)*(4*x[1]*x[1])"), degree=pdeg+1, a1=alpha_1, a2=alpha_2, lb = lbufr, rb = rbufr)
#pressure_constant = (- 1 * (lbufr + rbufr + right)**2 + 1/(3*reno)*(3*alpha_1 + alpha_2)*1**2*(lbufr + rbufr + right))

Analytic_Dq_1 = Expression(("-2", "r*2*(2*a1+a2)*(4*x[1]) - r*a1*4*x[1]"), degree=pdeg+1, a1=alpha_1, a2=alpha_2, r=reno, lb = lbufr, rb = rbufr)



#assign(P,project(q + alpha_1*dot(U,grad(q)),Q))
#vtkfile_P << P
assign(P,q)
vtkfile_P << P
if pipe:
    print('int_omega q = ', assemble(q*dx(mesh)) )
    

    q_base = assemble(Analytic_q/10*dx(mesh))
    print("q zero:", q_base)
    Analytic_q.q_base = q_base
    vtkfile_P << project(Analytic_q,Q1)
    vtkfile_P << project(q-Analytic_q,Q1)
    #vtkfile_P << project(project(Analytic_pressure,Q1)-P,Q)
    #vtkfile_P << project(-2-grad(q)[0],Q)
    #vtkfile_P << project(Analytic_Dq_1[1]-grad(q)[1],Q)
    
vtkfile_W << project(W,V2)
if pipe:
    vtkfile_W << project(WINFLOW,V2)
    vtkfile_W << project(W-WINFLOW,V)
    print('delta Qgradnorm: ',errornorm(Analytic_Dq_1, project(grad(q),V), norm_type='L2'))
    print('delta Wnorm: ', errornorm(WINFLOW,W, norm_type='L2'))
    print('delta Unorm: ', errornorm(UINFLOW,U, norm_type='H1'))
    print('delta Qnorm: ', errornorm(Analytic_q,q, norm_type='L2'))





U.vector().axpy(-1,nst.vector())
vtkfile_dU << project(U,V)
#vtkfile_dU << project(div(alpha_1*(A(u)*grad(u) + grad(u)*A(u)) + alpha_2*A(u)*A(u) alpha_1*(dot(u,nabla_grad(A(u))))
