#! /usr/bin/env python

"""
Extrapolation of correction parameters.
"""

__author__ = "Christian Waluga (waluga@ma.tum.de)"
__copyright__ = "Copyright (c) 2013 %s" % __author__

from dolfin import *
from correction import *
from meshtools import *
from singular import *
import math

def extrapolate_gamma_least_squares(h, g, angle):

  from scipy.optimize import leastsq
  p = 2.0 - 2.0*pi/angle
  fitfunc = lambda c, x: c[0] + c[1]*x**p
  errfunc = lambda c, x, y: (y - fitfunc(c, x))/x
  cinit = [g[-1] , 0.0, 0.0]
  c = leastsq(errfunc, cinit, args = (h, g), full_output = 1)
  return c[0][0], lambda x: fitfunc(c[0], x)


def extrapolate_gamma_romberg(h, g, angle):

  import numpy as np

  N = len(h)-1
  T = np.zeros((N,N))
  
  # compute Aitken-Neville tableau
  p = 2.0 - 2.0*pi/angle
  for i in range(0, N):
    T[i,0] = (h[i]**p * g[i+1] - h[i+1]**p * g[i])/(h[i]**p - h[i+1]**p)
  for i in range(1, N):
    for k in range(1, i+1):
      T[i,k] = T[i,k-1] + (T[i,k-1] - T[i-1,k-1])/((h[i-k]/h[i])**(p+1) - 1.0)

  return T[N-1,N-1], T


def extrapolate_gamma_richardson(h, g, angle):

  p = 2.0 - 2.0*pi/angle
  return g[-2] + (g[-1] - g[-2])/(1.0-(h[-1]/h[-2])**p)


def extrapolate_gamma(corner, angle, corner_mesh, func, method, maxit, \
                      refine_method, extrapolation, start_at, maxlevel, initial_gamma):

  if corner_mesh.size(2) is 0: return 0.0
  
  if refine_method == 'bulirsch':
  
    # refine meshes according to Bulirsch-series (1, 1/2, 1/3, 1/4, 1/6, 1/8, ...)
    meshes = [corner_mesh,refine(corner_mesh),refine3(corner_mesh)]
    
    for i in xrange(3, maxlevel):
      meshes.append(refine(meshes[-2]))

  elif refine_method == 'midpoint':

    # refine meshes by simple subdivision (1, 1/2, 1/4, 1/8, 1/16, ...)
    meshes = [corner_mesh]

    for i in xrange(1, maxlevel):
      meshes.append(refine(meshes[-1]))

  mesh = meshes[0]
  min_angle = find_min_angle(mesh, corner)

  # compute gammas using one-level algorithm
  if initial_gamma is None:
    initial_gamma = evaluate_fit(corner_mesh.size(2), angle, func == math.sin)
  g = compute_gammas(meshes, angle, min_angle, corner, initial_gamma = initial_gamma, \
                     maxit = maxit, func = func, method = method)
      
  import numpy as np
  
  h = [mesh.hmin() for mesh in meshes]

  x = np.asarray(h)
  y = np.asarray(g)

  if extrapolation == 'none':
    gamma_asymptotic = g[-1] # just use value computed on the highest level

  elif extrapolation == 'least-squares': # extrapolate by a least-squares fit
    gamma_asymptotic, fitfunc = extrapolate_gamma_least_squares(x[start_at:], y[start_at:], angle)
  
  elif extrapolation == 'romberg':
    gamma_asymptotic, tableau = extrapolate_gamma_romberg(x[start_at:], y[start_at:], angle)

  elif extrapolation == 'richardson':
    gamma_asymptotic = extrapolate_gamma_richardson(x[start_at:], y[start_at:], angle)

  # plot gamma
  if False: # just for debugging
    gammaa, fitfunc = extrapolate_gamma_least_squares(x[start_at:], y[start_at:], angle)

    import pylab
    fig = pylab.figure()
    plt, = pylab.semilogx(1./x, y, 'k*')
    xx = np.linspace(h[-1], h[0], 100)
    yy = fitfunc(xx)
    pylab.ylim((min(g)-0.05,max(g)+0.05))
    plt, = pylab.semilogx(1./xx, yy, 'r-')
    plt, = pylab.semilogx(1./xx, gamma_asymptotic*np.ones((len(xx),1)), 'b-')
    plt, = pylab.semilogx(1./xx, initial_gamma*np.ones((len(xx),1)), 'g-')
    pylab.savefig('output/gamma-{0}-{1}.pdf'.format(corner[0],corner[1]))

  return gamma_asymptotic, (h,g)


def extrapolate_gammas(corners, angles, corner_meshes, method = 'one-level-exact', maxit = 20, \
                       refine_method = 'bulirsch', extrapolation = 'least-squares', start_at = 3, \
                       maxlevel = 10, funcs = None, initial_gamma = None):

  g_asympt, data = [], []
  
  if funcs is None: # set all corners to Dirichlet by default
    funcs = [ math.sin for c in corners ]

  # for each corner, compute gamma
  for i in range(len(corners)):

    corner = corners[i]
    angle = angles[i]
    corner_mesh = corner_meshes[i]

    if method == 'fit':
      g, d = evaluate_fit(corner_mesh.size(2), angle, funcs[i] == math.sin), None
    else:
      g, d = extrapolate_gamma(corner, angle, corner_mesh, method = method, maxit = maxit, \
                               refine_method = refine_method, extrapolation = extrapolation, \
                               start_at = start_at, maxlevel = maxlevel, func = funcs[i], \
                               initial_gamma = initial_gamma)
    
    g_asympt.append(g)
    data.append(d)

  return g_asympt, data
