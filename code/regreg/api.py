"""
A collection of commonly used RegReg functions and objects
"""

# Atom imports

from .atoms import affine_atom as linear_atom
from .atoms.seminorms import (l1norm, l2norm, supnorm, 
                   positive_part, constrained_max,
                   constrained_positive_part, max_positive_part)
from .atoms.cones import (nonnegative, nonpositive,
                   zero, zero_constraint, 
                   l1_epigraph, l1_epigraph_polar,
                   l2_epigraph, l2_epigraph_polar,
                   linf_epigraph, linf_epigraph_polar,
                   affine_cone as linear_cone)
from .atoms.block_norms import l1_l2, linf_l2, l1_l1, linf_linf
from .atoms.svd_norms import (nuclear_norm, operator_norm,
                             nuclear_norm_epigraph,
                             operator_norm_epigraph)
from .atoms.linear_constraints import (projection, projection_complement)
from .atoms.mixed_lasso import mixed_lasso, mixed_lasso_conjugate
from .atoms.group_lasso import group_lasso, group_lasso_dual
from .atoms.weighted_atoms import (l1norm as weighted_l1norm,
                                   supnorm as weighted_supnorm)

# Affine imports

from affine import (identity, selector, affine_transform, normalize, linear_transform, composition as affine_composition, affine_sum,
                    power_L)
from affine.factored_matrix import (factored_matrix, compute_iterative_svd, soft_threshold_svd)

# Smooth imports

from smooth import (logistic_deviance, poisson_deviance, multinomial_deviance, smooth_atom, affine_smooth, logistic_loss, sum as smooth_sum)
from smooth.quadratic import quadratic, cholesky, signal_approximator, squared_error

# Problem imports

from problems.separable import separable, separable_problem
from problems.simple import simple_problem, gengrad, nesta, tfocs
from problems.container import container
from algorithms import FISTA
from problems.blocks import blockwise

from problems.conjugate import conjugate
from problems.composite import (composite, nonsmooth as nonsmooth_composite,
                       smooth as smooth_composite, smooth_conjugate)

from problems.dual_problem import dual_problem

from identity_quadratic import identity_quadratic

from paths import lasso, nesta as nesta_path, UNPENALIZED, L1_PENALTY, POSITIVE_PART, NONNEGATIVE

