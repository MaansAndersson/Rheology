""" Generate mesh """



from mshr import *
from dolfin import *
import math,sys
import numpy



dim = int(sys.argv[1])
meshsize= int(sys.argv[2])
design = str(sys.argv[3])
upright = 0.5
right = 1.
lbufr = -1; #float(sys.argv[3])
rbufr = 3; #float(sys.argv[4])

r0 = 0.5; #float(sys.argv[4])
r1 = 1; #float(sys.argv[4])

H = right

format = str(sys.argv[4])

if format == "pvd":
    mesh_file = File("mesh.pvd")
else :
    mesh_file = File("mesh.xml")

if design == "classic":
    if dim == 3 :
        cylinder1 = Cylinder(Point(lbufr, 0, 0), Point(0, 0, 0), 1.0, 1.0)
        cone = Cylinder(Point(0, 0, 0), Point(1, 0, 0), 1., upright)
        cylinder2 = Cylinder(Point(right, 0, 0), Point(right+rbufr, 0, 0), upright, upright)
        
        geometry = cone + cylinder1 + cylinder2
    elif dim == 2 :
        A = [dolfin.Point(lbufr, -1.0),\
        dolfin.Point( 0.0, -1.0),\
        dolfin.Point(right, -upright),\
        dolfin.Point(right+rbufr, -upright),\
        dolfin.Point(right+rbufr,  upright),\
        dolfin.Point(right,  upright),\
        dolfin.Point(0.0,  1.0),\
        dolfin.Point(lbufr,  1.0)]

        geometry = Polygon(A)
    mesh = generate_mesh(geometry,meshsize)
    
    
""" Geometry based on "" "" """
def NozzleRadius(z,H,r0,r1):
    return r0/sqrt(z/H*((r0/r1)**2-1)+1)
    
if design == "Göteborg":
   # H = 1.5
    L = 150
    r0 = 1
    r1 = 0.5
    #right = H
    H = right
    upright = r1; # NozzleRadius(right,H,r0,r1)

    if dim == 3 :
        print('foo')
    elif dim == 2 :
        ## Ugly code for creating Nozzle geometry
        # Note the parameter L regulates how fine the curvature is.
        MeshNodeList = [dolfin.Point(lbufr, -1.0),\
                        dolfin.Point( 0.0, -1.0)]
        for z in range(0,L):
            z *= H/L
            MeshNodeList.append(dolfin.Point(z, -NozzleRadius(z,H,r0,r1)))

        ## Adds buffer
        MeshNodeList.append(dolfin.Point(H, -r1))
        MeshNodeList.append(dolfin.Point(H+rbufr, -r1))
        MeshNodeList.append(dolfin.Point(H+rbufr, r1))
        MeshNodeList.append(dolfin.Point(H, r1))

        for z in range(0,L):
            z2 = H*(L-1-z)/L
            #print(z2)
            MeshNodeList.append(dolfin.Point(z2, NozzleRadius(z2,H,r0,r1)))


        MeshNodeList.append(dolfin.Point(0.0, 1.0))
        MeshNodeList.append(dolfin.Point(lbufr, 1.0))

        geometry = Polygon(MeshNodeList)
    mesh = generate_mesh(geometry,meshsize)

def NozzleRadius2(x,y,H,r0,r1):
#    if x > 0 and x < right :
    #return x,  r0*y/numpy.sqrt((right-x)/H*((pow(r0,2)/pow(r1,2))-1)+1)*(-1/(1-numpy.exp(-2*100*(x)))+1/(1-numpy.exp(-2*100*(x-right))))
     return x,  numpy.nan_to_num(r0*y/numpy.sqrt((right-x)/H*((pow(r0,2)/pow(r1,2))-1)+1)*numpy.bitwise_and(x > 0, x < right)) + ( x < 0 )*y + 0.5*( x > right )*y

def ClassicNozzle(x,y,H,r0,r1):
     return x,  (numpy.nan_to_num(-y/(1+x))*numpy.bitwise_and(x > 0, x < right)) + ( x < 0 )*y + 0.5*( x > right )*y
 #   else :
  #      return x, y

if design == 'pipe':
    mesh = RectangleMesh(Point(lbufr, -1), Point(right+rbufr, 1), 3*meshsize, meshsize, 'crossed')
    
if design == 'GöteborgCrossed':
    
    if dim == 3:
        print('foo')
        mesh0 = Mesh()
        #mesh = RectangleMesh(Point(lbufr, 0.5), Point(right+rbufr, 0), 32, 64, 'crossed')
        # transform 1/r
        # rotate 2pi
        
        mesh = RectangleMesh(Point(lbufr, -1), Point(right+rbufr, 1), 3*meshsize, meshsize, 'crossed')
        
        x = mesh.coordinates()[:,0]
        y = mesh.coordinates()[:,1]
        #z = mesh.coordinates()[:,1]
        
        #y_temp =
        #z_temp =
        print(mesh.cells())
            
        x_hat, y_hat = NozzleRadius2(x, y, H, r0, r1)
        
        print(min(x_hat))
        xyz_hat_coor = numpy.array([x_hat, y_hat]).transpose()
        mesh0.coordinates()[:] = xyz_hat_coor
        #mesh0.rotate(mesh0,20,2)
        mesh = mesh0
    elif dim == 2:
 
    
        #mesh0 = RectangleMesh(Point(lbufr, 1), Point(right, -1), 64, 64, 'crossed')
        mesh = RectangleMesh(Point(lbufr, -1), Point(right+rbufr, 1), 3*meshsize, meshsize, 'crossed')
        
        
        x = mesh.coordinates()[:,0]
        y = mesh.coordinates()[:,1]
            
        #x_hat, y_hat = x, y Density
        #xy_hat_coor = numpy.array([x_hat, y_hat]).transpose()
        #mesh.coordinates()[:] = xy_hat_coor
        
        
        
        x = mesh.coordinates()[:,0]
        y = mesh.coordinates()[:,1]
            
        x_hat, y_hat = NozzleRadius2(x, y, H, r0, r1)
        
        
        print(min(x_hat))
        xy_hat_coor = numpy.array([x_hat, y_hat]).transpose()
        mesh.coordinates()[:] = xy_hat_coor
        
       


# Refine
print(len(mesh.cells()))
for i in range(0,0):
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    for cell in cells(mesh):
      cell_markers[cell] = False
      p = cell.midpoint()
      if (abs(p.y())>0.4 and p.x() > 0.5):
          foo = 'foo'; #print("lol")# cell_markers[cell] = True
      if ((p.x() > rbufr-0.2) and (p.x() < lbufr+0.2)  ):
          cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)
print(len(mesh.cells()))



mesh_file << mesh
