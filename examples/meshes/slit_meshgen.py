#! /usr/bin/env python

"""
Mesh generator for domains with slits.
"""

__author__ = "Christian Waluga (waluga@ma.tum.de)"
__copyright__ = "Copyright (c) 2013 %s" % __author__


from dolfin import *
import numpy as np

mesh = RectangleMesh(-2.0, -1.0, 2.0, 1.0, 16, 8, 'crossed')

list_of_slits = []

def vertical_slit(X, Y):
  shift = np.array([1.0e-8, 0.0])
  on_slit  = lambda x: near(X, x[0]) and between(x[1], (Y[0], Y[1]))
  renumber = lambda x: x[0] > X
  return (on_slit, renumber, shift)

list_of_slits.append(vertical_slit(-1.5, (-1.0, -0.5 - 1e-4)))
list_of_slits.append(vertical_slit(-1.0, (0.0 + 1e-4, 1.0)))
list_of_slits.append(vertical_slit(-0.5, (-1.0, 0.0 - 1e-4)))
list_of_slits.append(vertical_slit(0.0, (0.5 + 1e-4, 1.0)))
list_of_slits.append(vertical_slit(0.5, (-1.0, 0.0 - 1e-4)))
list_of_slits.append(vertical_slit(1.0, (0.5 + 1e-4, 1.0)))
list_of_slits.append(vertical_slit(1.5, (-1.0, -0.5 - 1e-4)))

coord = mesh.coordinates().copy()
cells = mesh.cells().copy()

for (on_slit, renumber, shift) in list_of_slits:

  # find all nodes on the slit
  nc = len(coord)
  slit_vertices = [i for i in xrange(nc) if on_slit(coord[i])]

  # add new nodes
  import numpy as np
  newcoord = coord[slit_vertices]
  for i in xrange(len(newcoord)):
    newcoord[i,:] += shift
  coord = np.append(coord, newcoord, axis = 0)
  
  old_to_new = { }
  for i in xrange(len(slit_vertices)):
    old_to_new[slit_vertices[i]] = nc + i

  barycenter = lambda c: sum(c)/len(c)

  # split up mesh at the slit
  slit_vertices = set(slit_vertices)
  for i in xrange(len(cells)):
    c = cells[i]
    intersection = set(c).intersection(slit_vertices)
    if len(intersection) == 0 or not renumber(barycenter(coord[list(c)])):
      continue
    cells[i,:] = [old_to_new[c[j]] if c[j] in intersection else c[j] \
                                   for j in xrange(len(c))]

mesh = Mesh()
editor = MeshEditor()
editor.open(mesh, 2, 2);

# add vertices to mesh
editor.init_vertices(len(coord))
for i in xrange(len(coord)):
  editor.add_vertex(i, coord[i][0], coord[i][1])

# add cells to mesh
editor.init_cells(len(cells))
for i in xrange(len(cells)):
  editor.add_cell(i, cells[i,0], cells[i,1], cells[i,2])

# done: create and return mesh object
editor.close()
#mesh = refine(mesh)

filename = 'output/slit-channel-sym.xml'
File(filename) << mesh
from subprocess import call
call(['gzip', '-f', filename])

# test the mesh with finite element code

# function spaces
V = FunctionSpace(mesh, 'Lagrange', 1)

u = TrialFunction(V)
v = TestFunction(V)
problem = dot(grad(u),grad(v))*dx - 1.0*v*dx

bcs = DirichletBC(V, Expression('-x[0]'), 'on_boundary && (x[0] == -2 || x[0] == 2)')
uh = Function(V)
solve(lhs(problem) == rhs(problem), uh, bcs)

plot(uh)
interactive()








