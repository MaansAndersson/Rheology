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


alpha = 0.0
alpha_1 = Constant(alpha)  #0.001
alpha_2 = Constant(-alpha)  #-0.001
re = Constant(1.0) # Reynold's number
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
    uin = Expression(("(1.0-(x[1]*x[1]))","0"), degree = pdegV)
    u0 = Expression(("0","0"), degree = pdegV)
    p0 = Expression("0",degree = pdegQ)
    wi = Expression(("0","-0.5*8*x[1]*(3*a1+2*a2)"), a1 = alpha_1, a2 = alpha_2, degree = pdegV)


bcm1 = DirichletBC(W.sub(0), uin, InflowBoundary())
bcm2 = DirichletBC(W.sub(0), u0, NoSlipBoundary())
bcc = DirichletBC(W.sub(1), p0, OutFlowBoundary())
bcw = DirichletBC(V, wi, InflowBoundary())

bcm = [bcm2, bcm1]
bc = bcc

bcs = [bcm2, bcm1]

#bcc]

# TestFunctions
# v = TestFunction(V)
#q = TestFunction(Q)




# Functions
u = Function(V)

p = Function(Q)
p0 = Function(Q)

(v, q) = TestFunctions(W)
U = Function(W)
U0 = Function(W)

ust = Function(V)
#uvec_old = loadmat('ust')['uvec']
#ust.vector().set_local(uvec_old[:,0])
#u.vector().axpy(1, ust.vector())

#u.vector()[:] = 0

def A(z):
    return (grad(z) + grad(z).T)



(u, p) = (as_vector((U[0], U[1])), U[2]);
u0 = Function(V)

T = 1.0; t = 0; k = 0.001*(mesh.hmin())
print(k)
while (t < T):

    if True:
        um = 0.5*(u + u0)
        pm = 0.5*(p + p0)
    #for i in range(0,20):
    # Constitutive equations
        w = alpha_1*(1/k*(A(u)-A(u0)) + A(um)*grad(um) + grad(um)*A(um)) + alpha_2*A(um)*A(um)
        w += alpha_1*(dot(u,nabla_grad(A(u))))
        r = 1/re*inner(grad(um),grad(v))*dx + pm*div(v)*dx + inner(grad(um)*um,v)*dx - inner(div(w),v)*dx #- inner(div(sigma),v)*dx #+ inner(grad(u)*u,v)*dx
        r = -div(um)*q*dx
        r += h*h*inner(grad(pm),grad(q))*dx #+ h*h*inner(pm,q)*dx
        #r += ufl.Conditional(t<2*k,1,0)*h*h*inner(grad(u),grad(v))*dx
        
        
        #r += h*inner(alpha_1*(A(u)*grad(u) + grad(u)*A(u)),alpha_1*(A(v)*grad(v) + grad(v)*A(v)))*dx
        #r += h*inner(div(alpha_2*A(u)*A(u)),div(alpha_2*A(v)*A(v)))*dx
        #r +=  h*(inner(grad(pm) + grad(um)*um, grad(q) + grad(um)*v) + inner(div(um), div(v)))*dx #
        
#        if t<2*k :
 #
        
        if(MPI.rank(mesh.mpi_comm()) == 0):
            print('solve ')
        solve(r == 0, U, bcs, solver_parameters={"newton_solver":{"relative_tolerance":1e-10},"newton_solver":{"maximum_iterations":5}})
        t += k
        

    assign(u0, project(u,V))
    assign(p0, project(p,Q))
    vtkfile_u << u0 #project(u, V)#as_vector((w[0], w[1], w[2]))
    vtkfile_p << p0 #project(p, Q)#w[3]
    #vtkfile_s << project(w, V) # project(sigma,T)#as_matrix([[w[4+a+b] for b in range(0, 3)] for a in range(0, 3)])

