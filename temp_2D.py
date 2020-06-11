
from mshr import *
from dolfin import *
import math,sys

if format == "XML":
    mesh_file = File("mesh.xml")
else :
    mesh_file = File("mesh.pvd")

dim = int(sys.argv[1])
meshsize= int(sys.argv[2])
design = str(sys.argv[3])
lbufr = 1; #float(sys.argv[3])
rbufr = 4; #float(sys.argv[4])
r0 = 0.5; #float(sys.argv[4])
r1 = 1; #float(sys.argv[4])
H = 1.5
L = 150
upright = 0.5
right = 0.5


mesh = Mesh()
geometry = Polygon([dolfin.Point(lbufr, -1.0),\
   dolfin.Point( 0.0, -1.0),\
   dolfin.Point(right, -upright),\
   dolfin.Point(right+rbufr, -upright),\
   dolfin.Point(right+rbufr,  upright),\
   dolfin.Point(right,  upright),\
   dolfin.Point(0.0,  1.0),\
   dolfin.Point(lbufr,  1.0)])

#mesh = generate_mesh(geometry,meshsize)
PolygonalMeshGenerator.generate(mesh, geometry, 0.25);

mesh_file << mesh
