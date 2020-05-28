from mshr import *
from dolfin import *
import math,sys

from ufl import nabla_div
set_log_active(False)

mesh = Mesh("mesh.xml")

vtkfile_u = File('Ug2.pvd')
vtkfile_p = File('Pg2.pvd')
vtkfile_s = File('Sg2.pvd')

alpha_1 = Constant(0.01)  #0.001
alpha_2 = Constant(-0.01)  #-0.001
r = Constant(1.0)

V = VectorFunctionSpace(mesh, "Lagrange", 1)
Q = FunctionSpace(mesh, "Lagrange", 1)
T = TensorFunctionSpace(mesh, "Lagrange", 1)
h = CellDiameter(mesh)


class InflowBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and(x[0] <  -1 + DOLFIN_EPS)

class NoSlipBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0] > -1 + DOLFIN_EPS) and (x[0] < 4.5 - DOLFIN_EPS)

class OutFlowBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0] >= 4.5 - DOLFIN_EPS)


uin = Expression(("(1.0-(x[1]*x[1]+x[2]*x[2]))","0","0"), degree = 4)
u0 = Expression(("0","0","0"), degree = 1)
p0 = Expression("0",degree = 1)

bcm1 = DirichletBC(V, uin, InflowBoundary())
bcm2 = DirichletBC(V, u0, NoSlipBoundary())
bcc = DirichletBC(Q, p0, OutFlowBoundary())

bcm = [bcm2, bcm1]
bc = bcc


# TestFunctions
v = TestFunction(V)
q = TestFunction(Q)
tau = TestFunction(T)

# Functions
u = Function(V)
u0 = Function(V)
p = Function(Q)
p0 = Function(Q)
sigma = Function(T)
sigma0 = Function(T)

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

# Constitutive equations
rm = inner(grad(u),grad(v))*dx - p*div(v)*dx - inner(div(sigma),v)*dx
rc = div(u)*q*dx
rt = inner(sigma\
    + alpha_1*dot(u, nabla_grad(sigma)) \
    - (alpha_1*grad(u).T*A(u) \
    + (alpha_1 + alpha_2)*A(u)*A(u) \
    - r*outer(u,u) \
    - alpha_1*p*grad(u).T \
    + alpha_1*sigma0*grad(u).T),tau)*dx(mesh) \
    + alpha_1*0.5*div(u)*inner(sigma,tau)*dx(mesh) # Not divergance free.

# Stabilization
rm += h*r*inner(grad(u)*u,grad(v)*u)*dx
rc += h*h*inner(grad(p),grad(q))*dx
rt += 0.01*alpha_1*h*inner(dot(u,nabla_grad(sigma)), dot(u,nabla_grad(tau)))*dx(mesh) \
#      + 0.01*alpha_1*h*inner(nabla_grad(sigma),nabla_grad(tau))*dx(mesh)

for i in range(0,15):

    assign(p0, p)
    if(MPI.rank(mesh.mpi_comm()) == 0):
        print('solve pressure')
    solve(rc==0, p, bcc)
    
    assign(u0, u)
    if(MPI.rank(mesh.mpi_comm()) == 0):
        print('solve velocity')
    solve(rm==0, u, bcm)

    assign(sigma0, sigma)
    if(MPI.rank(mesh.mpi_comm()) == 0):
        print('solve stress')
    solve(rt==0, sigma)
    
    
    Uoldr.vector().axpy(-1, U.vector())
    relup = norm(Uoldr,'H1')
    relup /= norm(U,'H1')
    #relup = errornorm(u,u0,norm_type='H1',degree_rise=2)/norm(u,norm_type='H1')
    print("u update: ", relup)
    if relup < 1e-3: #(u0.vector().axpy(-1.0, u.vector()).norm("linf") / u.vector().norm("linf")) < 1e-4:
        print ("newton iteration: ", i, "Divergance: ", assemble(sqrt(div(u) * div(u))*dx)) #, " Rel change",
        break
    vtkfile_u << project(u,V)#as_vector((w[0], w[1], w[2]))
    vtkfile_p << project(p,Q)#w[3]
    vtkfile_s << project(sigma,T)#as_matrix([[w[4+a+b] for b in range(0, 3)] for a in range(0, 3)])
