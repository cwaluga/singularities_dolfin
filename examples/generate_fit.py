#! /usr/bin/env python

"""
Code used to generate the fits in Ruede/Waluga/Wohlmuth 2013
"""

__author__ = "Christian Waluga (waluga@ma.tum.de)"
__copyright__ = "Copyright (c) 2013 %s" % __author__

from dolfin import *
from energy_correction.meshtools import *
from energy_correction.extrapolate import *
import numpy as np
from math import pi
from scipy.optimize import leastsq

set_log_level(ERROR)

if __name__ == '__main__':
  
  maxlevel = 5
  nlayers0 = 2
  method = 'two-level-inexact'
  neumann = True
  
  nerange = range(3, 13)
  angles = np.linspace(1.1*pi, 2.0*pi, 60)
  
  corners = [np.asarray((0,0))]
  gammas = { }
  coeff = []

  for ne in nerange:
  
    gammas[ne] = [(pi, 0.0)]
    
    for angle in angles:

      # generate initial mesh
      mesh0 = generate_pie_mesh(ne, angle, nlayers0)
      
      # refine meshes according to Bulirsch-series
      meshes = [mesh0,refine(mesh0),refine3(mesh0)]
      for i in xrange(3, maxlevel):
        meshes.append(refine(meshes[-2]))

      funcs = [ (math.cos if neumann else math.sin) for c in corners ] # all Neumann/Dirichlet
      gamma = extrapolate_gammas(corners, [angle], meshes, method = method, \
                                 initial_gamma = gammas[ne][-1][1], \
                                 extrapolation = 'richardson', \
                                 maxlevel = maxlevel+2, funcs = funcs)[0][0]
      if gamma > 0:
        gammas[ne].append((angle, gamma))

      print ne, angle, gamma

    # do a nonlinear fit
    fitfunc = lambda c, x: c[0]*(np.exp(-2.0*(x - pi)) - 1.0) + c[1]*(x - pi)
    errfunc = lambda c, x, y: (y - fitfunc(c, x))*np.sqrt(x-pi+1e-3) # give less weight to the smaller angles

    cinit = [ 0.05, 0.10 ] # initial guess (chosen from numerical experiments)
    x = np.asarray([gammas[ne][i][0] for i in xrange(len(gammas[ne]))])
    y = np.asarray([gammas[ne][i][1] for i in xrange(len(gammas[ne]))])
    out = leastsq(errfunc, cinit, args = (x, y), full_output = 1)
    c = out[0]
    print c
    coeff.append([ne, c[0], c[1]])

  for c in coeff:
    print '{0}: [{1}, {2}],'.format(c[0], c[1], c[2])

