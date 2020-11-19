from mshr import *
from dolfin import *
import math,sys
import numpy as np
from petsc4py import PETSc
from scipy.io import loadmat, savemat
from timeit import default_timer as timer

#set_log_active(False)
set_log_active(True)

mesh = Mesh("mesh.xml")
dim = mesh.geometric_dimension()

mesh = refine(mesh)

mesh_out = File("mesh.xml")
mesh_out << mesh
