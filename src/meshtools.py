#! /usr/bin/env python

"""
Mesh generation, analysis and manipulation tools
"""

__author__ = "Christian Waluga (waluga@ma.tum.de)"
__copyright__ = "Copyright (c) 2013 %s" % __author__

from dolfin import * 

import math
import numpy as np

def find_reentrant_corners(mesh, eps = 1e-6):

  # get coordinates and cells of boundary mesh
  bmesh = BoundaryMesh(mesh, 'exterior', False)
  coords = bmesh.coordinates()
  cells = bmesh.cells()
  
  reentrant_corners = []; angles = []
  
  other = lambda vi, i: vi[0] if vi[1] == i else vi[1]
  normal = lambda x: np.array([x[1],-x[0]])
  
  for ic in xrange(len(coords)):
  
    # find two neighboring cells
    nc = [c for c in cells if ic in c]
    assert len(nc) is 2

    # compute tangential vectors
    t0 = coords[other(nc[0], ic)] - coords[ic]
    t1 = coords[other(nc[1], ic)] - coords[ic]

    # compute normal vectors
    n0 = normal(coords[nc[0][1]] - coords[nc[0][0]])
    n1 = normal(coords[nc[1][1]] - coords[nc[1][0]])
    
    # check if this is a reentrant corner
    # todo: sometimes wrong corners are found... check this
    #if (np.dot(t0,n1) > eps and np.dot(t1,n0) > eps) or np.dot(t0,t1) > 0:
    if (np.dot(t0,n1) > eps and (np.dot(t1,n0) > eps) or np.dot(t0,t1) > 0):
      reentrant_corners.append(coords[ic])
      angle = math.acos(np.dot(n0,n1)/math.sqrt(np.dot(n0,n0)*np.dot(n1,n1)))
      if angle < pi: angle += pi
      angles.append(angle)

  return np.asarray(reentrant_corners), np.asarray(angles)

def find_corner_elements(mesh, x):

  # find singular vertex in mesh
  for vertex in vertices(mesh):
    if all(vertex.x(i) == xi for i, xi in enumerate(x)):
      return vertex.entities(2)
  else: return [] # not found

def extract_corner_mesh(mesh, cells):

  marker = CellFunction("size_t", mesh)
  for i in cells: marker.set_value(i, 1)
  cornermesh = SubMesh(mesh, marker, 1)
  
  # generate parent_vertex_indices
  parent_vertex_indices = np.zeros(cornermesh.size(0), dtype='int')
  
  meshcoords = mesh.coordinates()
  cornercoords = cornermesh.coordinates()
  
  parent_vertices = set(x for x in mesh.cells()[cells].flatten())
  for i in parent_vertices:
    for j in xrange(len(cornercoords)):
      if np.linalg.norm(meshcoords[i] - cornercoords[j]) < 1e-12:
        parent_vertex_indices[j] = i

  # todo: use the following (once it is supported in the Python-frontend of dolfin)
  # cornermesh.data().array('parent_vertex_indices')
  
  return cornermesh, parent_vertex_indices


def symmetrize_corner_mesh(mesh, corner):
  
  # find limits
  mesh.init(1)
  
  if mesh.size(2) == 0: return mesh # needed for some meshes
  
  limits = []
  for vertex in vertices(mesh):
    if len(vertex.entities(2)) == 1:
      limits.append(vertex.index())
  assert len(limits) == 2

  # compute radii and angles
  x = mesh.coordinates()[:] - corner
  r = np.hypot(x[:,0], x[:,1])
  phi = np.arctan2(x[:,1], x[:,0])
  minphi = min([phi[l] for l in limits])
  maxphi = max([phi[l] for l in limits])
  if (maxphi - minphi) < pi:
    (maxphi, minphi) = (minphi+2.0*pi, maxphi)
  phi -= minphi
  phi[phi<0.0] += 2.0*pi

  # number of adjacent elements and angle of reentrant corner
  n = mesh.size(2)
  angle = (maxphi - minphi)

  # sort nodes
  phi_s = []
  for i in xrange(len(r)):
    if r[i] == 0.0: continue
    phi_s.append((i,phi[i]))
  phi_s = sorted(phi_s, key = lambda x: x[1])

  # compute new angles
  j = 0
  for (i,phi_i) in phi_s:
    phi[i] = minphi + j * angle / n
    j += 1

  # set radius of adjacent nodes to minimum
  r[r > 0] = sorted(r)[1]

  mesh.coordinates()[:,0] = r*np.cos(phi) + corner[0]
  mesh.coordinates()[:,1] = r*np.sin(phi) + corner[1]

  return mesh


# finds the minimum angle of a reentrant corner mesh
def find_min_angle(mesh, corner):

  mesh.init(1)
  limits = []
  for vertex in vertices(mesh):
    if len(vertex.entities(2)) == 1:
      limits.append(vertex.index())
  assert len(limits) == 2

  import numpy as np
  x = mesh.coordinates()[:] - corner
  r = np.hypot(x[:,0], x[:,1])
  phi = np.arctan2(x[:,1], x[:,0])
  minphi = min([phi[l] for l in limits])
  maxphi = max([phi[l] for l in limits])
  if (maxphi - minphi) < pi:
    (maxphi, minphi) = (minphi+2.0*pi, maxphi)
  return minphi-1e-12


# refines the mesh by dividing each triangle into 9 sub-triangles
def refine3(mesh):

  import numpy as np
  nv = mesh.coordinates()
  coords = np.zeros((10,2))
  
  id = 0
  vertices = {}
  elements = []
  as_tuple = lambda c: (c[0], c[1])
  
  for cell in cells(mesh):
  
    # create nodes
    coords[(0,3,9),:] = mesh.coordinates()[cell.entities(0),:]
    v1 = (coords[3]-coords[0])/3.0
    v2 = (coords[9]-coords[0])/3.0
    v3 = (coords[9]-coords[3])/3.0
    coords[1] = coords[0] + 1.0*v1
    coords[2] = coords[0] + 2.0*v1
    coords[4] = coords[0] + 1.0*v2
    coords[6] = coords[3] + 1.0*v3
    coords[7] = coords[0] + 2.0*v2
    coords[8] = coords[3] + 2.0*v3
    coords[5] = 0.5*(coords[4] + coords[6])
    
    ids = []
    for i in range(10):
      v = as_tuple(coords[i])
      if vertices.has_key(v):
        ids.append(vertices[v])
      else:
        ids.append(id); vertices[v] = id; id += 1
  
    map = lambda a, b, c: (ids[a],ids[b],ids[c])
    elements.append(map(0,4,1))
    elements.append(map(4,1,5))
    elements.append(map(1,5,2))
    elements.append(map(2,5,6))
    elements.append(map(6,3,2))
    elements.append(map(4,7,5))
    elements.append(map(7,5,8))
    elements.append(map(8,5,6))
    elements.append(map(7,8,9))

  # generate dolfin mesh
  mesh3 = Mesh()
  editor = MeshEditor()
  editor.open(mesh3, 2, 2);
  
  # add vertices to mesh
  editor.init_vertices(len(vertices))
  verts = vertices.items()
  for v in verts:
    editor.add_vertex(v[1], v[0][0], v[0][1])

  # add cells to mesh
  editor.init_cells(len(elements))
  id = 0
  for c in elements:
    editor.add_cell(id, c[0], c[1], c[2])
    id += 1

  # done: create and return mesh object
  editor.close()
  return mesh3


def generate_corner_info(mesh, symmetrization_threshold = 2.0*pi):

  # init mesh connectivity
  mesh.init(1)

  # find reentrant corners
  corners, angles = find_reentrant_corners(mesh)
  corner_meshes = []
  
  # extract corner meshes
  for i in xrange(len(corners)):
    
    angle = angles[i]
    
    elements = find_corner_elements(mesh, corners[i])
    corner_mesh, parent_vertex_indices = extract_corner_mesh(mesh, elements)
        
    if angle >= symmetrization_threshold:
        
      # make mesh symmetric
      corner_mesh = symmetrize_corner_mesh(corner_mesh, corners[i])
          
      # change elements also in original mesh
      mesh.coordinates()[parent_vertex_indices,:] = corner_mesh.coordinates()[:,:]

    corner_meshes.append(corner_mesh)

  return mesh, corners, angles, corner_meshes

def generate_pie_mesh(ne, omega, nlayers):

  import numpy
  from scipy.spatial import Delaunay
  from math import atan2, pi
  
  # hack: Delaunay-solution below does not like slit domain
  if abs(omega - 2.0*pi) < 1e-8: omega -= 1e-8
  
  helper = lambda n: 1 + n + ne * n * (n + 1)/2
  nvertices = helper(nlayers)
  vertices = numpy.zeros((nvertices, 2))
  
  # compute nodes for the different layers
  for i in range(0, nlayers):
    na = helper(i)
    nb = helper(i+1)
    angles = numpy.linspace (0.0, omega, nb - na)
    r = 1.0*(i+1)/nlayers
    vertices[na : nb, 0] = r * numpy.cos(angles)
    vertices[na : nb, 1] = r * numpy.sin(angles)
  
  # generate triangulation and filter unwanted elements
  tri = Delaunay(vertices)
  i = 0
  filter = []
  for ia, ib, ic in tri.vertices:
    # compute barycenter of triangle
    x = (vertices[ia] + vertices[ib] + vertices[ic])/3
    angle = atan2(x[1], x[0])
    if angle < 0: angle += 2.0*pi
    if angle < omega: filter.append(i)
    i += 1
  cells = tri.vertices[filter]

  # generate dolfin mesh
  mesh = Mesh()
  editor = MeshEditor()
  editor.open(mesh, 2, 2);
  
  # add vertices to mesh
  editor.init_vertices(len(vertices))
  id = 0
  for v in vertices:
    editor.add_vertex(id, v[0], v[1])
    id += 1

  # add cells to mesh
  editor.init_cells(len(cells))
  id = 0
  for c in cells:
    editor.add_cell(id, c[0], c[1], c[2])
    id += 1

  # done: create and return mesh object
  editor.close()
  return mesh


# parameters
plot_mesh = False
plot_corners = True

# unit test
if __name__ == '__main__':

  for filename in ['meshes/lshape.xml.gz', 'meshes/twocorners.xml.gz']:

    mesh = Mesh(filename)
    mesh, corners, angles, corner_meshes \
      = generate_corner_info(mesh, symmetrization_threshold = 1.5*pi)
    
    if plot_mesh: plot(mesh,interactive=True)
    
    if plot_corners:
    
      import matplotlib.pyplot as plt
      bmesh = BoundaryMesh(mesh, 'exterior')
      x = bmesh.coordinates()[:]
      plt.plot(x[:,0], x[:,1], marker='o', linestyle=' ', color='g')
      plt.plot(corners[:,0], corners[:,1], marker='o', linestyle=' ', color='r')
      plt.show()

    
