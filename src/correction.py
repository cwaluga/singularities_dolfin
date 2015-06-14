#! /usr/bin/env python

"""
Evaluation of fits and correction tools
"""

__author__ = "Christian Waluga (waluga@ma.tum.de)"
__copyright__ = "Copyright (c) 2013 %s" % __author__

from dolfin import *
from singular import *
from meshtools import *
import numpy as np


# evaluate fit which was obtained for p=1 and a symmetric mesh with isosceles
# cf. Ruede, Waluga, Wohlmuth 2013 for an explanation
def evaluate_fit(n, angle, dirichlet = True):

  fitfunc = lambda c, x: c[0]*(np.exp(-2.0*(x - pi)) - 1.0) + c[1]*(x - pi)

  if dirichlet:
    lookup = {\
      3: [0.0998183980437, 0.189615542703],
      4: [0.0555624819392, 0.128041557699],
      5: [0.0415019850858, 0.1072128902],
      6: [0.0363481781425, 0.0979881012415],
      7: [0.0328888599638, 0.0925971779024],
      8: [0.0313092655216, 0.0894450110842],
      9: [0.0304135897967, 0.0874557266743],
      10: [0.0289942470411, 0.0857622163158],
      11: [0.027901067039, 0.084485325609],
      12: [0.0279439846719, 0.0838604929991],
    }
  else:
    lookup = {\
      3: [0.165216485785, 0.212197384825],
      4: [0.0857378209084, 0.137467156594],
      5: [0.0560465904453, 0.111396584279],
      6: [0.044618021932, 0.10018816402],
      7: [0.038669805111, 0.0941769068565],
      8: [0.0352699798574, 0.0905576657623],
      9: [0.0326235916411, 0.0879857131768],
      10: [0.0311767624978, 0.0863397219782],
      11: [0.0292296008797, 0.0848041056068],
      12: [0.0282471423786, 0.0838033295999],
    }

  return fitfunc(lookup[n], angle)



# create a measure that allows us to assemble for the corner domains only
# this can be used to compute energies etc.
def corner_measure(mesh, corners):

  # init mesh connectivity
  mesh.init(1)

  # create mesh function with marked indices
  ncorners = len(corners)
  domain = CellFunction('size_t', mesh, ncorners)
  for vertex in vertices(mesh):
    for i in range(ncorners):
      if all(vertex.x(i) == x for i, x in enumerate(corners[i])):
        domain.array()[vertex.entities(2)] = i
        break

  return Measure('dx')[domain]


# defines the correction for the corner domains (probably slower than using varying coefficients)
def correction(u, v, gammas, dx_correction):

  corr = 0.0*dot(u,v)*dx
  for i in range(0,len(gammas)):
    corr -= Constant(gammas[i])*inner(grad(u), grad(v))*dx_correction(i)
  return corr


# correct the material parameter at the reentrant corners
def correct_corner_stiffness(function, mesh, corners, gammas):

  # init mesh connectivity
  mesh.init(1)

  # create mesh function with marked indices
  for vertex in vertices(mesh):
    for i in range(len(corners)):
      if all(vertex.x(i) == x for i, x in enumerate(corners[i])):
        function.vector()[vertex.entities(2)] *= (1.0-gammas[i])
        break

  return function


def solve_problem(mesh, corners, gammas, angles, u_exact = None):
  
  # right hand side
  f = Constant(1.0)

  # correct coefficient
  V0 = FunctionSpace(mesh, 'DG', 0)
  a = interpolate(Constant(1.0), V0)
  a = correct_corner_stiffness(a, mesh, corners, gammas)

  # variational form
  V = FunctionSpace(mesh, 'Lagrange', 1)
  u = TrialFunction(V)
  v = TestFunction(V)

  a = a*inner(grad(u), grad(v))*dx
  L = f*v*dx

  # boundary conditions
  bcs = [DirichletBC(V, 0.0, 'on_boundary')]

  # solve the variational problem
  uh = Function(V)
  solve(a == L, uh, bcs, \
          solver_parameters = { \
            "linear_solver": "bicgstab", \
            "preconditioner": "ml_amg"   \
          })

  if u_exact is None:
    return uh

  V_ho = FunctionSpace(mesh, 'Lagrange', 3)
  u_ho = interpolate(u_exact, V_ho)
  
  weight = WeightingFunction(corners, angles)
  return uh, sqrt(assemble(weight*(u_ho-uh)**2*dx))


# unit test
if __name__ == '__main__':

  from mpl_toolkits.mplot3d import Axes3D
  from matplotlib import cm
  from matplotlib.ticker import LinearLocator, FormatStrFormatter
  import matplotlib.pyplot as plt
  import numpy as np
  
  import pylab
  pylab.rc("font", family = "serif")
  pylab.rc("font", size = 12)

  # generate first plot
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  X = np.linspace(3, 12, 10)
  Y = np.linspace(1.0, 2.0-1e-3, 10)
  X, Y = np.meshgrid(X, Y)
  Z = np.vectorize(lambda X, Y: evaluate_fit(X, Y*pi, dirichlet = True))(X, Y)
  surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.binary, # choose cm.Spectral for color
          linewidth=.5, antialiased=True)
  ax.set_zlim(0.0, 0.5)

  ax.set_xlabel(r'$n$')
  ax.set_ylabel(r'$\theta/\pi$')
  ax.set_zlabel(r'$\gamma_{\infty}$')
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

  fig.colorbar(surf, shrink=0.5, aspect=5)

  fig.savefig('gammafit3d.pdf')
  plt.show()

  # generate second plot

  X = np.linspace(3, 7, 5)
  Y = np.linspace(1.0, 2.0-1e-3, 20)
  
  fig = plt.figure()

  plots, labels = [], []
  for i,n in enumerate(X):
    p, = plt.plot(Y, evaluate_fit(n, Y*pi), ['kv-', 'g^-', 'rD-', 'bs-', 'y*-'][i])
    plots.append(p)
    labels.append('n={0}'.format(n))
  
  plt.legend(plots,labels,loc=0)
  plt.xlabel(r'$\theta/\pi$')
  plt.ylabel(r'$\gamma_\infty$')

  fig.savefig('gamma-plot-angle-n=3-7.pdf')
  plt.show()

