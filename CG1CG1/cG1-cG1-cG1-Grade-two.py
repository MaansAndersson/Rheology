from mshr import *
from dolfin import *
import math,sys
from scipy.io import loadmat, savemat
from ufl import nabla_div
set_log_active(True)

mesh = Mesh("mesh.xml")
mesh = (refine(mesh))
dim = mesh.geometric_dimension()

vtkfile_u = File('Ug_t.pvd')
vtkfile_p = File('Pg_t.pvd')
#vtkfile_s = File('Sg2.pvd')

alpha_1 = Constant(0.01)  #0.001
alpha_2 = Constant(-0.01)  #-0.001
r = Constant(1.0) # Reynold's number
# Samma

pdegV = 1
pdegQ = 1

V = VectorFunctionSpace(mesh, "Lagrange", pdegV)
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
    uin = Expression(("(1.0-(x[1]*x[1]+x[2]*x[2]))","0","0"), degree = pdegV)
    u0 = Expression(("0","0","0"), degree = pdegV)
    p0 = Expression("0",degree = pdegQ)
else:
    uin = Expression(("exp(-1000*(-1-x[0])*(-1-x[0]))*(1.0-(x[1]*x[1]))","0"), degree = pdegV)
    u0 = Expression(("0","0"), degree = pdegV)
    p0 = Expression("x[1]*x[1]",degree = pdegQ)
    wi = Expression(("0","-0.5*8*x[1]*(3*a1+2*a2)"), a1 = alpha_1, a2 = alpha_2, degree = pdegV)


bcm1 = DirichletBC(W.sub(0), uin, InflowBoundary())
bcm2 = DirichletBC(W.sub(0), u0, NoSlipBoundary())
bcc = DirichletBC(W.sub(1), p0, OutFlowBoundary())
bcw = DirichletBC(V, wi, InflowBoundary())

bcm = [bcm2, bcm1]
bc = bcc

bcs = [bcm2, bcm1] #, bcc]

# TestFunctions
# v = TestFunction(V)
#q = TestFunction(Q)




# Functions
u = Function(V)

p = Function(Q)
p0 = Function(Q)
sigma = Function(T)
sigma0 = Function(T)

w = Function(V)
w_ = TrialFunction(V)
tau = TestFunction(V)

#(u_, p_) = TrialFunctions(W)
(v, q) = TestFunctions(W)
U = Function(W)
U0 = Function(W)

ust = Function(V)
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



(u, p) = (as_vector((U[0], U[1])), U[2]);
u0 = Function(V)

T = 1.0; t = 0; k = (0.5*mesh.hmin())
print(k)
while (t < T):

    if True:
        um = 0.5*(u + u0)
        pm = 0.5*(p + p0)
    #for i in range(0,20):
    # Constitutive equations
        w = alpha_1*(1/k*(A(u)-A(u0)) + A(um)*grad(um) + grad(um)*A(um)) + alpha_2*A(um)*A(um)
        r = 1/r*inner(grad(um),grad(v))*dx + pm*div(v)*dx + inner(grad(um)*um,v)*dx + inner(div(w),v)*dx #- inner(div(sigma),v)*dx #+ inner(grad(u)*u,v)*dx
        r = -div(um)*q*dx
        #rc += h*h*inner(grad(pm),grad(q))*dx
        r += h*h*(inner(grad(pm) + grad(um)*um, grad(q) + grad(um)*v) + inner(div(um), div(v)))*dx
        
        if t<2*k:
            r += h*inner(grad(u),grad(v))*dx
        
        if(MPI.rank(mesh.mpi_comm()) == 0):
            print('solve ')
        solve(r == 0, U, bcs, solver_parameters={"newton_solver": {"relative_tolerance": 1e-12}})
        t += k
        

        U0.vector().axpy(-1, U.vector())
        relup = norm(U0.vector(),'linf')
        relup /= norm(U.vector(),'linf')
        #relup = errornorm(u,u0,norm_type='H1',degree_rise=2)/norm(u,norm_type='H1')
        if(MPI.rank(mesh.mpi_comm()) == 0):
            print("u update: ", relup)
        if relup < 1e-8 and i > 3: #(u0.vector().axpy(-1.0, u.vector()).norm("linf") / u.vector().norm("linf")) < 1e-4:
            if(MPI.rank(mesh.mpi_comm()) == 0):
                print ("newton iteration: ", i, "Divergance: ", assemble(sqrt(div(u) * div(u))*dx)) #, " Rel change",
            break
    assign(u0, project(u,V))
    assign(p0, project(p,Q))
    vtkfile_u << u0 #project(u, V)#as_vector((w[0], w[1], w[2]))
    vtkfile_p << p0 #project(p, Q)#w[3]
    #vtkfile_s << project(w, V) # project(sigma,T)#as_matrix([[w[4+a+b] for b in range(0, 3)] for a in range(0, 3)])

