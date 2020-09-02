from mshr import *
from dolfin import *
import math,sys
from scipy.io import loadmat, savemat
from ufl import nabla_div
from timeit import default_timer as timer
set_log_active(False)

pdeg = 4
fudg = 10000*1000
reno = float(sys.argv[1])
alfa = float(sys.argv[2]) #0.01
lbufr = -1; #float(sys.argv[3])
rbufr = 3; #float(sys.argv[4])
r0 = 0.5; #float(sys.argv[4])
r1 = 1; #float(sys.argv[4])
right = 1.0 #0.5



pipe = str(sys.argv[3])
if pipe == 'pipe':
    pipe = True
else:
    pipe = False
    
if pipe:
    upright = 1. #.5 #0.5 #0.5
else:
    upright = 0.5

alpha_1 = Constant(alfa)
alpha_2 = Constant(-alfa)
realf = reno*alfa

mesh = Mesh("mesh.xml")
dim = mesh.geometric_dimension()
print(dim)
cell = mesh.ufl_cell()
print("pdeg: ", pdeg)


Vp = VectorFunctionSpace(mesh, "Lagrange", pdeg)
# Refine inlet and outlet
for i in range(0,3):
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    for cell in cells(mesh):
      cell_markers[cell] = False
      p = cell.midpoint()
      if (p.x()-1.0)**2 + (p.y()-0.5)**2 < 0.1*0.1 or (p.x()-1.0)**2 + (p.y()-0.5)**2 < 0.1*0.1:
          cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)
    
for i in range(0,1):
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    for cell in cells(mesh):
      cell_markers[cell] = False
      p = cell.midpoint()
      if abs(p.y()-0) > 0.49 and (p.x()-1)>:
          cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)
print(len(mesh.cells()))



vtkfile_U = File('results/g2u_2D.pvd')
vtkfile_P = File('results/g2p_2D.pvd')
vtkfile_W = File('results/g2W_2D.pvd')
vtkfile_dU = File('results/grade2W-nse_2D.pvd')

Bubble = False
if Bubble:
    # some demands on pdeg.
    V1 = FiniteElement("Lagrange", mesh.ufl_cell(), pdeg)
    B = FiniteElement("B", mesh.ufl_cell(), mesh.topology().dim() + 1)
    V = FunctionSpace(mesh, VectorElement(NodalEnrichedElement(V1, B)))
    V2 = VectorFunctionSpace(mesh, "CG", pdeg-3)
    Q = FunctionSpace(mesh, "Lagrange", pdeg-1)
else:
    V = VectorFunctionSpace(mesh, "Lagrange", pdeg)
    Q = FunctionSpace(mesh, "Lagrange", pdeg-1)
    Q1 = FunctionSpace(mesh, "Lagrange", pdeg)
    V2 = VectorFunctionSpace(mesh, "CG", pdeg)
    Z = FunctionSpace(mesh, "Lagrange", pdeg-1)

def in_bdry(x):
    return x[0] <= lbufr+DOLFIN_EPS

# define boundary condition
if dim == 2 :
    boundary_exp = Expression(("exp(-fu*(lb-x[0])*(lb-x[0]))*(1.0-x[1]*x[1]) + \
      (1.0/up)*exp(-fu*(ri+rb-x[0])*(ri+rb-x[0]))*(1.0-((x[1]*x[1])/(up*up)))","0"), \
                       up=upright,ri=right,fu=fudg,rb=rbufr,lb=lbufr,degree = pdeg)
                       
    boundary_zee = Expression("exp(-fu*(lb-x[0])*(lb-x[0]))*(2.0*x[1]) + \
               exp(-fu*(ri+rb-x[0])*(ri+rb-x[0]))*(2.0*x[1]/(up*up*up))", \
                       up=upright,ri=right,fu=fudg,rb=rbufr,lb=lbufr,degree = pdeg)
    #boundary_zee = Expression("0.0")

else :
    boundary_exp = Expression(("exp(-fu*(lb-x[0])*(lb-x[0]))*(1.0-(x[1]*x[1]+x[2]*x[2])) + \
      (1.0/up)*exp(-fu*(ri+rb-x[0])*(ri+rb-x[0]))*(1.0-((x[1]*x[1]+x[2]*x[2])/(up*up)))","0","0"), \
                       up=upright,ri=right,fu=fudg,rb=rbufr,lb=lbufr,degree = pdeg)
    

bc = DirichletBC(V, boundary_exp, "on_boundary")
bczee = DirichletBC(Z, boundary_zee, in_bdry)

# set the parameters
r = Constant(0*1/1.0)  # time-step artificial compressability
rp = Constant(0*200.0) # step-length artificial compressability
ro = Constant(1.e4)   # penelty in penelty iteration



U = project(Function(Vp,'nst.xml'),V)
nst = Function(V)
nst.vector()[:] = U.vector()[:]
U = project(Function(Vp,'ust.xml'),V)
ust = Function(V)
ust.vector()[:] = U.vector()[:]
P = Function(Q)

v = TestFunction(V)



h = CellDiameter(mesh)

#Pre-define variables
incrnorm = 1; gtol = alfa*1e-5 #alfa*0.0000001; #alfa*0.0001;  r = 1.0e4;
n = FacetNormal(mesh)

def A(z):
    return (grad(z) + grad(z).T)

gold = TrialFunction(V)
asg = (inner(grad(gold),grad(v))             \
#      +reno*zee*(v[0]*gold[1]-v[1]*gold[0]) \
       +ro*div(gold)*div(v))*dx
gold = Function(V)
gold.vector().axpy(1.0, nst.vector())
#gold.vector().axpy(1.0, uz.vector())
# Grade-two solver
gters = 0; max_gters = 20; incrgoldnorm = 1
errfn = Function(V)
goldr = Function(V)
gtol=alfa*0.0001
while gters < max_gters and incrgoldnorm > gtol:
    wg = Function(V)
    zee = Function(Z)
#   zee = project((grad(gold[1])[0]-grad(gold[0])[1]),Z)
    dub = TestFunction(Z)
#   zform = (zee*dub + realf*inner(gold,grad(zee))*dub)*dx
    ZF =(zee*dub + realf*inner(gold,grad(zee))*dub-(grad(gold[1])[0]-grad(gold[0])[1])*dub)*dx
#   zF = (grad(gold[1])[0]-grad(gold[0])[1])*dub*dx
    solve(ZF == 0, zee, bczee)
    FF = reno*zee*(v[0]*gold[1]-v[1]*gold[0])*dx
    bg = -div(wg)*div(v)*dx
    pdegt = LinearVariationalProblem(asg, FF + bg, gold, bc)
    solverg = LinearVariationalSolver(pdegt)
    iters = 0; max_iters = 5; div_u_norm = 1
    #wg=project(zf,V)
    while iters < max_iters and div_u_norm > 1e-10:
    # solve and update wg
        solverg.solve()
        wg.vector().axpy(ro, gold.vector())
    # find the L^2 norm of div(u) to check stopping condition
        div_u_norm = sqrt(assemble(div(gold)*div(gold)*dx(mesh)))
        print( "   IPM iter_no=",iters,"div_u_norm="," %.5e"%div_u_norm )
        iters += 1
    #plot(div(uold), interactive=True)
    print( "   IPM iter_no=",iters,"div_u_norm="," %.2e"%div_u_norm)
    incrgoldnorm=errornorm(gold,goldr,norm_type='H1',degree_rise=2)/norm(gold,norm_type='H1')
#   incrgoldnorm=errornorm(gold,nst,norm_type='H1',degree_rise=4)/norm(gold,norm_type='H1')
#   incrgoldnorm=norm(errfn,norm_type='H1')/norm(gold,norm_type='H1')
    print( gters,"change="," %.3e"%incrgoldnorm)
    gters += 1
#   print norm(goldr,norm_type='H1')
    goldr = Function(V)
#   print norm(goldr,norm_type='H1')
    goldr.vector().axpy(1, gold.vector())
#   print norm(goldr,norm_type='H1')
#   plot(goldr[0], interactive=True)
#plot(gold[0], interactive=True)
# compare Stokes with grade-two

"""
goldnorm=norm(ust,norm_type='H1')
stomingtoonorm=errornorm(gold,ust,norm_type='H1')/goldnorm
stominavstorm=errornorm(nst,ust,norm_type='H1')/goldnorm
"""


# compare Navier-Stokes with grade-two
gold.vector().axpy(-1.0, nst.vector())
#U.vector().axpy(-1,nst.vector())
#gold.vector().axpy(-1.0, nst.vector())
vtkfile_dU << project(gold,V)
goldnstnorm=norm(gold,norm_type='H1')/goldnorm


#vtkfile_U << project(U,V)


# For a 2D pipe
Analytic_pressure = Expression(( "-2*((x[0]-1.5)) + (2*a1+a2)*(4*x[1]*x[1])"), degree=pdeg+1, a1=alpha_1, a2=alpha_2, lb = lbufr, rb = rbufr)

Analytic_Dq_1 = Expression(("-2", "2*(2*a1+a2)*(4*x[1]) - a1*4*x[1]"), degree=pdeg+1, a1=alpha_1, a2=alpha_2, lb = lbufr, rb = rbufr)



#assign(P,project(q + alpha_1*dot(U,grad(q)),Q))
#vtkfile_P << P
#assign(P,q)
#vtkfile_P << P
"""
if pipe:
    vtkfile_P << project(Analytic_pressure,Q1)
    vtkfile_P << project(project(Analytic_pressure,Q1)-P,Q)
    vtkfile_P << project(-2-grad(q)[0],Q)
    vtkfile_P << project(Analytic_Dq_1[1]-grad(q)[1],Q)
vtkfile_W << project(W,V2)
if pipe:
    vtkfile_W << project(WINFLOW,V2)
    vtkfile_W << project(W-WINFLOW,V)
    print('delta Qgradnorm: ',errornorm(Analytic_Dq_1, project(grad(q),V), norm_type='L2'))
    print('delta Wnorm: ', errornorm(WINFLOW,W, norm_type='L2'))
"""



