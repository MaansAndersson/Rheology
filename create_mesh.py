""" Generate mesh """



from mshr import *
from dolfin import *
import math,sys



dim = int(sys.argv[1])
meshsize= int(sys.argv[2])
design = "classic" #str(sys.argv[2])
lbufr = 1; #float(sys.argv[3])
rbufr = 4; #float(sys.argv[4])
r0 = 0.5; #float(sys.argv[4])
r1 = 1; #float(sys.argv[4])
upright = 0.5
right = 0.5


format = "XML"

if format == "XML":
    mesh_file = File("mesh.xml")
else :
    mesh_file = File("mesh.pvd")

if design == "classic":
    if dim == 3 :
        cylinder1 = Cylinder(Point(-lbufr, 0, 0), Point(0, 0, 0), 1.0, 1.0)
        cone = Cylinder(Point(0, 0, 0), Point(1, 0, 0), 1., upright)
        cylinder2 = Cylinder(Point(right, 0, 0), Point(right+rbufr, 0, 0), upright, upright)
        
        geometry = cone + cylinder1 + cylinder2
    elif dim == 2 :
        geometry = [dolfin.Point(lbufr, -1.0),\
           dolfin.Point( 0.0, -1.0),\
           dolfin.Point(right, -upright),\
           dolfin.Point(right+rbufr, -upright),\
           dolfin.Point(right+rbufr,  upright),\
           dolfin.Point(right,  upright),\
           dolfin.Point(0.0,  1.0),\
           dolfin.Point(lbufr,  1.0)]
    
    
    
""" Geometry based on "" "" """
def NozzleRadius(z,H,r0,r1):
    return r0/sqrt(z/H*((r0/r1)**2-1)+1)
    
if type == "Göteborg":
    if dim == 3 :
        print('foo')
    elif dim == 2 :
        ## Ugly code for creating Nozzle geometry
        # Note the parameter L regulates how fine the curvature is.
        if domain_bool == 'NewNozzle':
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


if type == "pipe":
    
    if dim == 3:
        print('foo')
        mesh = RectangleMesh(Point(lbufr, 0.5), Point(right+rbufr, 0), 15, 64, 'crossed')
        # transform 1/r
        # rotate 2pi
    elif dim == 2:
        mesh = RectangleMesh(Point(lbufr, 1), Point(right+rbufr, -1), 15, 64, 'crossed')



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
