#! /usr/bin/env python

"""
Definition of the singularities, energy computation and Newton algorithms 
"""

__author__ = "Christian Waluga (waluga@ma.tum.de)"
__copyright__ = "Copyright (c) 2013 %s" % __author__

from dolfin import *
import math


# transform coordinates from cartesian to polar
def transform(x, x0, phi0):

  from math import hypot, atan2
  X = x - x0
  r = hypot(X[0], X[1])
  phi = atan2(X[1], X[0]) - phi0
  if phi < 0: phi = phi + 2*pi
  return r, phi


# boundary condition for the pie-meshes
def boundary(x, on_boundary, dirichlet, origin, angle, min_angle, epsilon = 1e-4):
  if dirichlet: return on_boundary
  r, phi = transform(x, origin, min_angle)
  return on_boundary and min(abs(phi), abs(angle - phi)) > epsilon


# first singularity function
class SingularityFunction(Expression):

  def __init__(self, a, x0, phi0, func): # set to cos for Neumann
    self.a, self.x0, self.phi0, self.func = a, x0, phi0, func
  def eval(self, values, x):
    r, phi = transform(x, self.x0, self.phi0)
    values[0] = r**(self.a)*self.func(self.a*phi)
    return


# weighting function for multiple corners
class WeightingFunction(Expression):

  def __init__(self, corners, angles, factor = 1.0):
    self.corners, self.angles, self.factor = corners, angles, factor
  
  def eval(self, values, x):
    values[0] = 1.0
    v = []
    for i in xrange(len(self.corners)):
      X = x - self.corners[i]
      v.append(self.factor*(X[0]**2 + X[1]**2)**(1.0-math.pi/self.angles[i]))
    values[0] = min(min(v), 1.0)
    return


# solve the finite element problem with energy-correction
def compute_energies(mesh, angle, min_angle, corner, gamma, func):

  from correction import corner_measure
  
  # create linear finite element space
  V = FunctionSpace(mesh, 'Lagrange', 1)
  u = TrialFunction(V)
  v = TestFunction(V)

  # assemble weak problem with energy correction
  dx_corner = corner_measure(mesh, [corner])
  problem = inner(grad(u), grad(v))*dx \
          - Constant(gamma)*inner(grad(u), grad(v))*dx_corner(0)

  # boundary conditions
  s = SingularityFunction(pi/angle, corner, min_angle, func = func)
  dirichlet = True#(func == math.sin) # use Dirichlet problem for Neumann as well
  bf = lambda x, on_boundary: \
       boundary(x, on_boundary, dirichlet, corner, angle, min_angle)
  bc = DirichletBC(V, s, bf)
  #plot(bc)

  # assemble and solve the linear system
  uh = Function(V)
  solver_parameters = {
    'linear_solver': 'bicgstab',
    'preconditioner': 'ml_amg'
  }
  solve(lhs(problem) == rhs(problem), uh, bc, solver_parameters = solver_parameters)

  return [assemble(inner(grad(uh),grad(uh))*dx), \
          assemble(inner(grad(uh),grad(uh))*dx_corner(0))]


# compute the exact energy a(s,s) of the singular function
def compute_exact_energy(mesh, angle, min_angle, corner, func, k = 3):

  W = FunctionSpace(mesh, 'Lagrange', k)
  a = pi/angle
  s = SingularityFunction(a, corner, min_angle, func = func)
  sk = interpolate(s, W)
  
  class CornerBoundary(SubDomain):
    def inside(self, x, on_boundary):
      r, phi = transform(x, corner, min_angle)
      return on_boundary and min(abs(phi), abs(angle - phi)) < 1e-4

  corner_boundary = CornerBoundary()
  marker = FacetFunction('size_t', mesh)
  marker.set_all(0)
  corner_boundary.mark(marker, 1)
  ds = Measure('ds')[marker]

  # we use an integration by parts here to avoid adaptive integration near the singularity:
  # there holds -div(grad(u) = 0 inside the domain and u = 0 at the boundaries touching the singularity.
  meas = assemble(Constant(1.0)*ds(1), mesh = mesh)
  en = assemble(Dn(sk)*sk*ds(0)) + (0.0 if (func == math.sin) else  (meas**(a + 1.0)/(a + 1.0)))
  return en


# the Newton algorithms of Ruede/Waluga/Wohlmuth 2013
def compute_gammas(meshes, angle, min_angle, corner, func, \
                   initial_gamma, method, maxit, tol = 1e-8, use_previous = True):

  gammas = [initial_gamma]
  
  limiter = lambda g: min(0.5, max(0.0,g))

  if method == 'one-level-inexact':
  
    e0 = compute_energies(meshes[0], angle, min_angle, corner, gammas[0], func = func)
    
    for mesh in meshes[1:]:

      e1 = compute_energies(mesh, angle, min_angle, corner, gammas[-1], func = func)
      gamma = limiter((e1[0] - e0[0])/(e1[1] - e0[1]))
      gammas.append(gamma)
      e0 = e1

  elif method == 'two-level-inexact':
    
    oldmesh = meshes[0]
    for mesh in meshes[1:]:

      e0 = compute_energies(oldmesh, angle, min_angle, corner, gammas[-1], func = func)
      e1 = compute_energies(mesh, angle, min_angle, corner, gammas[-1], func = func)
      gamma = limiter((e1[0] - e0[0])/(e1[1] - e0[1]))
      gammas.append(gamma)
      oldmesh = mesh

  elif method == 'one-level-exact':
  
    if func == math.cos:
      print 'warning: exact Newton is not yet properly implemented for Neumann problems'

    for mesh in meshes[0:]:
      
      gamma = gammas[-1] if use_previous else initial_gamma
      ass = compute_exact_energy(mesh, angle, min_angle, corner, func = func)
      
      # check if k0=1 is suitable
      #a, c = compute_energies(mesh, angle, min_angle, corner, 0.5, func = func)
      #print 'k0=1 is good' if ass - a + 0.5*c > 0.0 else 'k0=1 is bad'
      
      i, res = 0, float('inf')
      TOL = (tol if type(tol) is float else tol(mesh, ass))
      while i < maxit and res > TOL:
      
        a, c = compute_energies(mesh, angle, min_angle, corner, gamma, func = func)
        res = abs(ass - a + gamma*c)
        gamma = limiter((a - ass) / c)
        i += 1
      #print i, gamma
      gammas.append(gamma)

    gammas = gammas[1:]
    
  else:
    print 'error: method \'%s\' not implemented' % method

  return gammas


# unit test
if __name__ == '__main__':

  from meshtools import *
  
  symmetrization_threshold = 1.5*pi

  mesh, corners, angles, corner_meshes \
    = generate_corner_info(Mesh('meshes/twocorners.xml.gz'), symmetrization_threshold)

  maxlevel = 10

  #plot(mesh)

  set_log_level(ERROR)

  for i in range(len(corners)):
    corner = corners[i]
    angle = angles[i]
    corner_mesh = corner_meshes[i]
    
    if corner_mesh.size(2) is 0: continue;
    
    # refine meshes according to Bulirsch-series
    meshes = [corner_mesh,refine(corner_mesh),refine3(corner_mesh)]
    for i in xrange(3, maxlevel):
      meshes.append(refine(meshes[-2]))

    mesh = meshes[0]
    min_angle = find_min_angle(mesh, corner)

    # compute gammas using one-level algorithm
    initial_gamma = evaluate_fit(corner_mesh.size(2), angle)
    g = compute_gammas(meshes, angle, min_angle, corner, initial_gamma, func = func)

    import numpy as np
    
    # compute Aitken-Neville tableau
    h = [mesh.hmin() for mesh in meshes]
    h = h[2:]
    g = g[2:]
  
    #print 'h =', h, '\ngamma =', g,
    
    N = len(h)-1
    T = np.zeros((N,N))
    
    p = 2.0 - 2.0*pi/angle
    for i in range(0, N):
      T[i,0] = (h[i]**p * g[i+1] - h[i+1]**p * g[i])/(h[i]**p - h[i+1]**p)
    for i in range(1, N):
      for k in range(1, i+1):
        T[i,k] = T[i,k-1] + (T[i,k-1] - T[i-1,k-1])/((h[i-k]/h[i])**(p+1) - 1.0)

    # output
    #np.set_printoptions(precision = 6)
    #print '\nAitken-Neville table:\n', T
    print '\nasymptotic value of gamma:', T[N-1,N-1], 'fit:', evaluate_fit(corner_mesh.size(2), angle)

  interactive()



  
