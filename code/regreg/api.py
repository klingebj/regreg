"""
A collection of commonly used RegReg functions and objects
"""

from atoms import (l1norm, l2norm, supnorm, 
                   positive_part, constrained_max, affine_atom as linear_atom,
                   constrained_positive_part, max_positive_part)
from cones import (nonnegative, nonpositive,
                   zero, zero_constraint, 
                   l1_epigraph, l1_epigraph_polar,
                   l2_epigraph, l2_epigraph_polar,
                   linf_epigraph, linf_epigraph_polar,
                   affine_cone as linear_cone)

from linear_constraints import (projection, projection_complement)

from affine import (identity, selector, affine_transform, normalize, linear_transform, composition as affine_composition, affine_sum,
                    power_L)
from smooth import (logistic_deviance, poisson_deviance, multinomial_deviance, smooth_atom, affine_smooth, logistic_loss, sum as smooth_sum)
from quadratic import quadratic, cholesky, signal_approximator, squared_error

from factored_matrix import (factored_matrix, compute_iterative_svd, soft_threshold_svd)

from separable import separable, separable_problem
from simple import simple_problem, gengrad, nesta, tfocs
from container import container
from algorithms import FISTA
from admm import admm_problem
from blocks import blockwise

from block_norms import l1_l2, linf_l2, l1_l1, linf_linf

from conjugate import conjugate
from composite import (composite, nonsmooth as nonsmooth_composite,
                       smooth as smooth_composite, smooth_conjugate)

from dual_problem import dual_problem

from identity_quadratic import identity_quadratic

from weighted_atoms import (l1norm as weighted_l1norm,
                            supnorm as weighted_supnorm)

from paths import lasso
