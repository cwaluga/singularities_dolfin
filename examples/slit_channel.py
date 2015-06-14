#! /usr/bin/env python
# coding=utf-8

"""
Example from the paper Rüde/Waluga/Wohlmuth 2013
"""

__author__ = "Christian Waluga (waluga@ma.tum.de)"
__copyright__ = "Copyright (c) 2013 %s" % __author__


from dolfin import *
from energy_correction.correction import *
from energy_correction.meshtools import *
from energy_correction.singular import *
from energy_correction.extrapolate import *
import math

set_log_level(ERROR)


def solve_problem(mesh, corners, gammas):

  # right hand side
  f = Constant(0.0)

  # correct coefficient
  V0 = FunctionSpace(mesh, 'DG', 0)
  k = interpolate(Constant(1.0), V0)
  k = correct_corner_stiffness(k, mesh, corners, gammas)
  
  #plot(k)

  # variational form
  V = FunctionSpace(mesh, 'Lagrange', 1)
  u = TrialFunction(V)
  v = TestFunction(V)

  a = k*inner(grad(u), grad(v))*dx
  L = f*v*dx

  # boundary conditions
  #bcs = [ DirichletBC(V, Expression('1.0', c = 0.25), 'on_boundary && (x[0] == -2.0)'), \
  #        DirichletBC(V, Expression('0.0', c = 0.25), 'on_boundary && (x[0] == +2.0)')  ]
  bcs = [ DirichletBC(V, Expression('1.0+c*cos(pi*x[1])', c = 0.25), 'on_boundary && (x[0] == -2.0)'), \
          DirichletBC(V, Expression('0.0+c*cos(pi*x[1])', c = 0.25), 'on_boundary && (x[0] == +2.0)')  ]


  # solve the variational problem
  uh = Function(V)
  print 'problem size:', V.dim()
  solver_parameters = { \
    "linear_solver": "bicgstab", \
    "preconditioner": "petsc_amg" }
  
  solve(a == L, uh, bcs, solver_parameters = solver_parameters)
  return uh


# convergence rates
def print_rates(E):
  from math import log as lg
  print   '%.4e   -    %.4e   -    ' \
    %(E[0][0], E[0][1])
  for i in range(1, len(E)):
    r1 = lg(E[i-1][0]/E[i][0], 2)
    r2 = lg(E[i-1][1]/E[i][1], 2)
    print '%.4e  %.2f  %.4e  %.2f' \
      %(E[i][0], r1, E[i][1], r2)
  print '\n'


# boundary definition for error
class RightBoundary(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(x[0], 2.0)


if __name__ == '__main__':

  # main program
  parameters["allow_extrapolation"] = True

  maxlevel = 7    # maximum level (this is where 'exact' solution is computed)
  compute = False  # does the 'exact' solution need to be computed?
  gzip = True     # gzip the 'exact' solution once computed and saved?
  correct = False  # do we want energy-correction
  
  mesh_filename = 'meshes/slit-channel-crossed.xml.gz'
  
  cachedgamma = False # use cached values of gamma or recompute with method specified below?
  method = 'one-level-inexact'
  #method = 'two-level-inexact'

  # beyond the L-shape, we have to symmetrize if we want to use only one correction per corner.
  # (set to 2pi if the mesh is already symmetric at the corners)
  symmetrization_threshold = 2.0*pi
  
  output_weight = False # output the weighting function to VTK?

  # find reentrant corners
  mesh, corners, angles, corner_meshes \
    = generate_corner_info(Mesh(mesh_filename), symmetrization_threshold)

  weight = WeightingFunction(corners, angles, 4.0)

  #plot(mesh); interactive()
  
  gammas_cached = [0.27914627419934601, 0.27914440403310525, 0.27916755164797358, 0.27912214602624991, 0.27908848023107002, 0.27915828177500074, 0.2791861111329178]

  if correct:
    if cachedgamma:
      print 'using cached gammas'
      gammas = gammas_cached
    else:
      print 'computing gammas'
      funcs = [ math.cos for c in corners ] # all Neumann
      gammas = extrapolate_gammas(corners, angles, corner_meshes, method = method, \
                                  start_at = maxlevel, extrapolation = 'richardson', \
                                  maxlevel = maxlevel+2, funcs = funcs)[0]
  else:
    gammas = [0.0 for i in range(len(corners))]
  print 'gammas =', gammas

  # generate series of refined meshes
  print 'generating meshes'
  meshes = [ mesh ]
  for i in xrange(maxlevel):
    meshes.append(refine(meshes[-1]))

  if output_weight:
    File('output/weight.pvd') << interpolate(weight, FunctionSpace(meshes[3], 'Lagrange', 1))

  filename = 'output/uh_fine_{0}'.format(maxlevel)
  if compute:
    print 'computing fine solution'
    if not correct: print 'warning: computing fine solution without correction'
    uh_fine = solve_problem(meshes[-1], corners, gammas)
    File(filename + '.xml') << uh_fine
    File(filename + '.pvd') << uh_fine
    if gzip:
      from subprocess import call
      call(['gzip', '-f', filename + '.xml'])
  else:
    print 'loading fine solution'
    V_fine = FunctionSpace(meshes[-1], 'Lagrange', 1)
    uh_fine = Function(V_fine, filename + '.xml.gz' if gzip else '')

  errors = []

  print 'solving'

  # perform a convergence study
  for i, mesh in enumerate(meshes[:-2]):

    uh = solve_problem(mesh, corners, gammas)
    intorder, intmesh = 1, meshes[i+2]
    U = project(uh_fine, FunctionSpace(intmesh, 'Lagrange', intorder))
    
    File('output/uh-{0}.pvd'.format(i)) << uh
    
    right_boundary = RightBoundary()
    boundaries = FacetFunction("size_t", intmesh)
    boundaries.set_all(0)
    right_boundary.mark(boundaries, 1)
    dG = Measure("ds")[boundaries]

    #h = CellSize(mesh)
    h = Constant(mesh.hmin())
    
    # compute errors between current level and highest level solutions
    err1 = sqrt(assemble((uh - U)**2*dx, mesh = intmesh))
    err2 = sqrt(assemble(weight*(uh - U)**2*dx, mesh = intmesh))
    #err2 = sqrt(assemble(h*(Dn(uh) - Dn(U))**2*dG(1), mesh = intmesh))
    errors.append((err1, err2))
    print_rates(errors)

