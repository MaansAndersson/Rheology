from mshr import *
from dolfin import *
import math,sys
from scipy.io import loadmat, savemat
from ufl import nabla_div
set_log_active(False)

mesh = Mesh("mesh.xml")

for i in range(4):
    mesh = (refine(mesh)) # refine(refine(refine(refine(mesh))))
dim = mesh.geometric_dimension()

vtkfile_u = File('Ug2.pvd')
vtkfile_p = File('Pg2.pvd')
vtkfile_s = File('Sg2.pvd')


alfa = 0.01 
alpha_1 = Constant(alfa)  #0.001
alpha_2 = Constant(-alfa)  
r = Constant(1) # Reynold's number
# Samma

pdegV = 1
pdegQ = 1

V = VectorFunctionSpace(mesh, "Lagrange", pdegV)
V2 = VectorFunctionSpace(mesh, "CG", pdegV)
Q = FunctionSpace(mesh, "Lagrange", pdegQ)
Z = FunctionSpace(mesh, "DG", pdegQ)

VE = VectorElement("Lagrange", mesh.ufl_cell(), pdegV)
QE = FiniteElement("Lagrange", mesh.ufl_cell(), pdegQ)
W = FunctionSpace(mesh, MixedElement([VE,QE]))


T = TensorFunctionSpace(mesh, "Lagrange", 1)
h = CellDiameter(mesh)


class InflowBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0] <  -1 + DOLFIN_EPS)

class NoSlipBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0] > -1 + DOLFIN_EPS) and (x[0] < 4 - DOLFIN_EPS)

class OutFlowBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0] > 4 - DOLFIN_EPS)

if dim == 3:
    uin = Expression(("(1.0-(x[1]*x[1]+x[2]*x[2]))","0","0"), degree = pdegV+2)
    boundary_exp = Expression(("exp(-fu*(lb-x[0])*(lb-x[0]))*(1.0-x[1]*x[1]) + \
      (1.0/up)*exp(-fu*(ri+rb-x[0])*(ri+rb-x[0]))*(1.0-((x[1]*x[1])/(up*up)))","0"), \
                       up=upright,ri=right,fu=fudg,rb=rbufr,lb=lbufr,degree = pdeg+2)

    u0 = Expression(("0","0","0"), degree = pdegV)
    p0 = Expression("0",degree = pdegQ)
else:
    lbufr = -1
    rbufr = 3
    upright = 0.5
    right = 1

    
    uin = Expression(("exp(-1000*1000*(-1-x[0])*(-1-x[0]))*(1.0-(x[1]*x[1]))","0"), degree = pdegV)
    boundary_exp = Expression(("exp(-fu*(lb-x[0])*(lb-x[0]))*(1.0-x[1]*x[1]) + \
      (1.0/up)*exp(-fu*(ri+rb-x[0])*(ri+rb-x[0]))*(1.0-((x[1]*x[1])/(up*up)))","0"), \
                       up=upright,ri=right,fu=1000,rb=rbufr,lb=lbufr,degree = 2)

    u0 = Expression(("0","0"), degree = pdegV)
    p0 = Expression("0",degree = pdegQ)
    wi = Expression(("0","-0.5*8*x[1]*(3*a1+2*a2)"), a1 = alpha_1, a2 = alpha_2, degree = pdegV)


bcm1 = DirichletBC(W.sub(0), uin, InflowBoundary())
bcm2 = DirichletBC(W.sub(0), u0, NoSlipBoundary())
bcm3 = DirichletBC(W.sub(0), boundary_exp, 'on_boundary')
bcc = DirichletBC(W.sub(1), p0, OutFlowBoundary())
#bcw = DirichletBC(V, wi, InflowBoundary())


#bcm = [bcm2, bcm1]
bc = bcc

bcs = bcm3 #
#bcs = [bcm2, bcm1, bcc]

# TestFunctions
# v = TestFunction(V)
#q = TestFunction(Q)




# Functions
u = Function(V)
n = FacetNormal(mesh)
p = Function(Q)
p0 = Function(Q)
sigma = Function(T)
sigma0 = Function(T)

w = Function(V)
w_ = TrialFunction(V)
tau = TestFunction(V)

(u_, p_) = TrialFunctions(W)
(v, q) = TestFunctions(W)
U = Function(W)
u0 = Function(W)

u_out = Function(V)
w_out = Function(V2)
p_out = Function(Q)

ust = Function(V)
pst = Function(Q)
#uvec_old = loadmat('ust')['uvec']
#ust.vector().set_local(uvec_old[:,0])
#u.vector().axpy(1, ust.vector())


# Trial Functions
#w = Function(W)
#(u, p, sigma) = (as_vector((w[0], w[1], w[2])), w[3], as_matrix([[w[4+a+b] for b in range(0, 3)] for a in range(0, 3)])) # w[2])

#w0 = Function(W)
#(u0, p0, sigma0) = (as_vector((w0[0], w0[1], w0[2])), w0[3], as_matrix([[w0[4+a+b] for b in range(0, 3)] for a in range(0, 3)])) #, w0[2])


#u.vector()[:] = 0

def A(z):
    return (grad(z) + grad(z).T)



if True:
    um = u

rm = inner(grad(u_),grad(v))*dx - p_*div(v)*dx
rc = div(u_)*q*dx
rc += h*h*inner(grad(p_),grad(q))*dx + h*h*h*p_*q*dx
rm += inner(ust,v)*dx
rc += inner(pst,q)*dx


a = lhs(rm+rc)
L = rhs(rm+rc)

Am, b = assemble_system(a, L, bcs)
solve(Am, U.vector(), b, 'lu')

#solve(a == L, U, bcs)
ust_, pst_ = U.split()

assign(ust, ust_)
assign(pst, pst_)
assign(u_out, project(ust, V))
vtkfile_u << u_out    

for i in range(0,20):
# Constitutive equations
    rm = inner(grad(u_),grad(v))*dx(mesh) - p_*div(v)*dx(mesh) - inner(w,v)*dx #- inner(div(sigma),v)*dx #+ inner(grad(u)*u,v)*dx
    rc = div(u_)*q*dx

    """
    rt = inner(sigma\
        + alpha_1*dot(u, nabla_grad(sigma)) \
        - (alpha_1*grad(u).T*A(u) \
        + (alpha_1 + alpha_2)*A(u)*A(u) \
        - r*outer(u,u) \
        - alpha_1*p*grad(u).T \
        + alpha_1*sigma0*grad(u).T),tau)*dx(mesh) \
        + alpha_1*0.5*div(u)*inner(sigma,tau)*dx(mesh) # Not divergance free.
    """


    # Stabilization
    rm += Constant(1)*h*inner(grad(u_)*u,grad(v)*u)*dx
    rc += h*h*inner(grad(p_),grad(q))*dx + h*h*h*p_*q*dx 




    assign(u0, U)
    #assign(p0, p)

    
    if(MPI.rank(mesh.mpi_comm()) == 0):
        print('solve ')
    a = lhs(rm+rc)
    L = rhs(rm+rc)
    
    Am, b = assemble_system(a, L, bcs)
    """for bc in bcs:
        bc.apply(Am)
        bc.apply(b)"""
    
    solve(Am, U.vector(), b, 'lu')
    
    #solve(a == L, U, bcs)
    u, p = U.split()
    
    """
    rz =  inner(1/r*w_ \
    + alpha_1*dot(u, nabla_grad(w_)) \
    - div(alpha_1*grad(u).T*A(u) \
    + (alpha_1 + alpha_2)*A(u)*A(u) \
    - r*outer(u,u) \
    - alpha_1*p*grad(u).T), tau)*dx(mesh) \
    + Constant(0)*h*inner(grad(w_), grad(tau))*dx(mesh) \
    + Constant(0.1)*alpha_1*h*inner(dot(u,nabla_grad(w_)), dot(u,nabla_grad(tau)))*dx(mesh)
    #+ 1*abs(alpha_1*dot(u('+'),n('+')))*conditional(dot(u('+'),n('+'))<0,1,0)*inner(jump(w_),tau('+'))*dS(mesh)
    """
    
    rz =  inner(1/r*w_ \
     + alpha_1*dot(u, nabla_grad(w_)) \
     - div(alpha_1*grad(u).T*A(u) \
     + (alpha_1 + alpha_2)*A(u)*A(u) \
     - outer(u,u) \
     + alpha_1*grad(u).T*p), tau)*dx(mesh) \
     + Constant(0.0)*alpha_1*h*inner(grad(w_), grad(tau))*dx(mesh) \
     + Constant(1)*alpha_1*h*inner(dot(u,nabla_grad(w_)), dot(u,nabla_grad(tau)))*dx(mesh) 
    


    #assign(sigma0, sigma)
    if(MPI.rank(mesh.mpi_comm()) == 0):
        print('solve stress')
    #solve(rz == 0, w, bcw)
    aa = lhs(rz)
    bb = rhs(rz)
    solve(aa == bb, w) #, bcw)
   # w.vector()[:] = 0   
 

    u0.vector().axpy(-1, U.vector())
    relup = norm(u0.vector(),'linf')
    relup /= norm(U.vector(),'linf')
    #relup = errornorm(u,u0,norm_type='H1',degree_rise=2)/norm(u,norm_type='H1')
    if(MPI.rank(mesh.mpi_comm()) == 0):
        print("u update: ", relup)
    if relup < 1e-8 and i > 3: #(u0.vector().axpy(-1.0, u.vector()).norm("linf") / u.vector().norm("linf")) < 1e-4:
        if(MPI.rank(mesh.mpi_comm()) == 0):
            print ("newton iteration: ", i, "Divergance: ", assemble(sqrt(div(u) * div(u))*dx)) #, " Rel change",
        break
    u, p = U.split()
assign(u_out, project(u, V))
assign(w_out, project(w, V))
assign(p_out, project(p, Q))
vtkfile_u << u_out#as_vector((w[0], w[1], w[2]))
vtkfile_p << p_out#w[3]
vtkfile_s << w_out # project(sigma,T)#as_matrix([[w[4+a+b] for b in range(0, 3)] for a in range(0, 3)])
assign(u_out, project(u-ust, V))
assign(p_out, project(p-pst, Q))
vtkfile_u << u_out#as_vector((w[0], w[1], w[2]))
vtkfile_p << p_out#w[3]
vtkfile_s << w_out
