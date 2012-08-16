from warnings import warn
import gc

import numpy as np
import scipy.sparse

from .affine import power_L, normalize, selector, identity, adjoint
from .atoms import l1norm, constrained_positive_part
from .smooth import logistic_loss, sum as smooth_sum, affine_smooth
from .quadratic import squared_error
from .separable import separable_problem, separable
from .simple import simple_problem
from .identity_quadratic import identity_quadratic as iq

# Constants used below

UNPENALIZED = -1
L1_PENALTY = -2
POSITIVE_PART = -3

class lasso(object):

    def __init__(self, loss_factory, X, elastic_net=iq(0,0,0,0),
                 alpha=0., intercept=True,
                 positive_part=None,
                 unpenalized=None,
                 lagrange_proportion = 0.05,
                 nstep = 100,
                 scale=True,
                 center=True):
        self.loss_factory = loss_factory

        self.scale = scale
        self.center = center

        print 'positive_part', positive_part
        # normalize X, adding intercept if needed
        self.intercept = intercept
        if self.intercept:
            p = X.shape[1]

            if scipy.sparse.issparse(X):
                self._X1 = scipy.sparse.hstack([np.ones((X.shape[0], 1)), X]).tocsc() 
            else:
                self._X1 = np.hstack([np.ones((X.shape[0], 1)), X])
            if self.scale or self.center:
                self._Xn = normalize(self._X1, center=self.center, scale=self.scale, intercept_column=0)
            else:
                self._Xn = self._X1

            if unpenalized is None:
                unpenalized = np.zeros(p, np.bool)
                unpenalized[0] = 1
            else:
                unpenalized_b = np.zeros(p+1, np.bool)
                unpenalized_b[np.arange(1,p+1)[unpenalized]] = 1
                unpenalized_b[0] = 1
                unpenalized = unpenalized_b

            if positive_part is None:
                positive_part = np.zeros(p, np.bool)
            else:
                positive_part_b = np.zeros(p+1, np.bool)
                positive_part_b[np.arange(1,p+1)[positive_part]] = 1
                positive_part = positive_part_b

        else:
            if self.scale or self.center:
                self._Xn = normalize(X, center=self.center, scale=self.scale)
            else:
                self._Xn = X

            if unpenalized is None:
                unpenalized = np.zeros(p, np.bool)
            else:
                unpenalized_b = np.zeros(p, np.bool)
                unpenalized_b[unpenalized] = 1
                unpenalized = unpenalized_b

            if positive_part is None:
                positive_part = np.zeros(p, np.bool)
            else:
                positive_part_b = np.zeros(p, np.bool)
                positive_part_b[positive_part] = 1
                positive_part = positive_part_b


        which_0 = self._Xn.col_stds == 0
        if np.any(which_0):
            self._selector = selector(~which_0, self._Xn.primal_shape)
            if self.scale or self.center:
                self._Xn = self._Xn.slice_columns(~which_0)
            else:
                self._Xn = self._Xn[:,~which_0]
        else:
            self._selector = identity(self._Xn.primal_shape)

        # the penalty parameters
        self.alpha = alpha
        self.lagrange_proportion = lagrange_proportion
        self.nstep = nstep
        self._elastic_net = elastic_net.collapsed()

        # settle what is penalized and what is not
        p = self.shape[1]
        self.penalty_structure = np.zeros(p, np.int)
        self.penalty_structure[:] = L1_PENALTY
        if self.intercept:
            self.penalty_structure[0] = UNPENALIZED
        self.penalty_structure[positive_part] = POSITIVE_PART
        self.penalty_structure[unpenalized] = UNPENALIZED

        print self.penalty_structure, positive_part, 'again'
        if not (np.all(self.penalty_structure[positive_part] == POSITIVE_PART) and
                np.all(self.penalty_structure[unpenalized] == UNPENALIZED)):
            warn('conflict in positive part and unpenalized coefficients')

        self.ever_active = self.penalty_structure == UNPENALIZED
        grad_solution = np.zeros(p)


    @property
    def shape(self):
        if self.scale or self.center:
            return self.Xn.dual_shape[0], self.Xn.primal_shape[0]
        else:
            return self.Xn.shape

    @property
    def nonzero(self):
        return self._selector

    @property
    def elastic_net(self):
        q = self._elastic_net
        q.coef *= self.lagrange
        q.linear_term *= self.lagrange
        return q

    @property
    def Xn(self):
        return self._Xn

    @property
    def loss(self):
        if not hasattr(self, '_loss'):
            self._loss = self.loss_factory(self._Xn)
        return self._loss

    @property
    def null_solution(self):
        if not hasattr(self, "_null_soln"):
            n, p = self.shape
            self._null_soln = np.zeros(p)
            null_problem, null_selector = self.restricted_problem(self.penalty_structure == UNPENALIZED, self.lagrange_max)[:2]
            self._null_soln = null_selector.adjoint_map(null_problem.solve())
        return self._null_soln

    @property
    def lagrange_max(self):
        if not hasattr(self, "_lagrange_max"):
            null_soln = self.null_solution
            null_grad = self.loss.smooth_objective(null_soln, 'grad')
            l1_set = self.penalty_structure == L1_PENALTY
            if l1_set.sum():
                l1_lagrange_max = np.fabs(null_grad)[l1_set].max()
            else:
                l1_lagrange_max = -np.inf

            pp_set = self.penalty_structure == POSITIVE_PART
            if pp_set.sum():
                pp_lagrange_max = null_grad[pp_set].max()
            else:
                pp_lagrange_max = -np.inf
            self._lagrange_max = max(l1_lagrange_max, pp_lagrange_max)
        return self._lagrange_max

    def get_lagrange_sequence(self):
        if not hasattr(self, "_lagrange_sequence"):
            self._lagrange_sequence = self.lagrange_max * np.exp(np.linspace(np.log(self.lagrange_proportion), 0, 
                                                                             self.nstep))[::-1]
        return self._lagrange_sequence

    def set_lagrange_sequence(self, lagrange_sequence):
        self._lagrange_sequence = lagrange_sequence
    
    lagrange_sequence = property(get_lagrange_sequence, set_lagrange_sequence)

    @property
    def problem(self):
        p = self.shape[1]
        if not hasattr(self, "_problem"):
            self._problem = self.restricted_problem(np.ones(self.shape[1], np.bool),
                                                    self.lagrange_max)[0]
        return self._problem

    # for now, the lagrange of the positive part is the same as the L1 so it
    # can be found on either of the atoms..
    def get_lagrange(self):
        return self._problem.proximal_atom.atoms[0].lagrange

    def set_lagrange(self, lagrange):
        proximal_atom = self._problem.proximal_atom
        for atom, group in zip(proximal_atom.atoms, proximal_atom.groups):
            atom.lagrange = lagrange
            atom.quadratic = self.elastic_net[group]
    lagrange = property(get_lagrange, set_lagrange)

    @property
    def solution(self):
        return self.problem.coefs

    @property
    def active(self):
        return self.solution != 0

    @property
    def lipschitz(self):
        if not hasattr(self, "_lipschitz"):
            self._lipschitz = power_L(self.Xn)
        return self._lipschitz

    def grad(self, loss=None):
        '''
        Gradient at current value. This includes the gradient
        of the smooth loss as well as the gradient of the elastic net part.
        This is used for determining whether the KKT conditions are met
        and which coefficients are in the strong set.
        '''
        if loss is None:
            loss = self.loss
        gsmooth = self.loss.smooth_objective(self.solution, 'grad')
        penalized = self.penalty_structure != UNPENALIZED
        gquad = self.elastic_net.objective(self.solution[penalized], 'grad')

        gsmooth[penalized] += gquad
        return gsmooth

    def strong_set(self, lagrange_cur, lagrange_new, 
                   slope_estimate=1, grad=None):
        if grad is None:
            grad = self.grad()

        if (hasattr(self, 'dual_term') and self.dual_term is not None and not np.all(self.dual_term == 0)):
            grad += self.nesta_term.affine_transform.adjoint_map(self.dual_term) # sign correct?

        value = np.zeros(grad.shape, np.bool)

        # strong set for l1 penalty
        l1_set = self.penalty_structure == L1_PENALTY
        if l1_set.sum():
            value[l1_set] += np.fabs(grad[l1_set]) < (slope_estimate+1) * lagrange_new - slope_estimate*lagrange_cur

        # strong set for constrained_positive_part penalty

        # lagrange multipler of pospart is the same as the l1 for now

        pp_set = self.penalty_structure == POSITIVE_PART
        if pp_set.sum():
            value[pp_set] += -grad[pp_set] < (slope_estimate+1) * lagrange_new - slope_estimate*lagrange_cur

        value[self.ever_active] = False
        value = ~value
        p = self.shape[1]
        return value, selector(value, (p,))

    def slice_columns(self, columns):
        if self.scale or self.center:
            Xslice = self.Xn.slice_columns(columns)
        else:
            Xslice = self.Xn[:,columns]
        return Xslice

    def restricted_problem(self, candidate_set, lagrange):
        '''
        Assumes the candidate set includes intercept as first column.
        '''

        Xslice = self.slice_columns(candidate_set)

        loss = self.loss_factory(Xslice)
        restricted_penalty_structure = self.penalty_structure[candidate_set]
        rps = restricted_penalty_structure # shorthand
        if self.intercept:
            Xslice.intercept_column = 0
        l1_set = rps == L1_PENALTY
        pp_set = rps == POSITIVE_PART

        groups = [l1_set, pp_set]
        penalties = [penalty(group.sum(), lagrange=lagrange) for group, penalty 
                     in zip([l1_set, pp_set], [l1norm, constrained_positive_part])]

        for penalty, group in zip(penalties, groups):
            penalty.quadratic = self._elastic_net[group]

        penalty_sliced = separable(Xslice.primal_shape, penalties, groups)
        problem_sliced = simple_problem(loss, penalty_sliced)
        candidate_selector = selector(candidate_set, self.Xn.primal_shape)
        return problem_sliced, candidate_selector, restricted_penalty_structure

    def solve_subproblem(self, candidate_set, lagrange_new, **solve_args):
    
        # try to solve the problem with the active set
        subproblem, selector, penalty_structure = self.restricted_problem(candidate_set, lagrange_new)
        subproblem.coefs[:] = selector.linear_map(self.solution)
        sub_soln = subproblem.solve(**solve_args)
        self.solution[:] = selector.adjoint_map(sub_soln)

        grad = subproblem.smooth_objective(sub_soln, mode='grad') 
        self.final_inv_step = subproblem.final_inv_step
        return self.final_inv_step, grad, sub_soln, penalty_structure

    def main(self, inner_tol=1.e-5):

        # scaling will be needed to get coefficients on original scale   
        if self.Xn.scale:
            scalings = np.asarray(self.Xn.col_stds).reshape(-1)
        else:
            scalings = np.ones(self.Xn.primal_shape)
        scalings = self.nonzero.adjoint_map(scalings)

        # take a guess at the inverse step size
        self.final_inv_step = self.lipschitz / 1000
        lseq = self.lagrange_sequence # shorthand

        # first solution corresponding to all zeros except intercept 

        self.solution[:] = self.null_solution.copy()

        grad_solution = self.grad().copy()
        strong, strong_selector = self.strong_set(lseq[0], lseq[1], grad=grad_solution)

        p = self.shape[0]

        rescaled_solutions = scipy.sparse.csr_matrix(self.nonzero.adjoint_map(self.solution) 
                                                     / scalings)

        objective = [self.loss.smooth_objective(self.solution, 'func')]
        dfs = [np.sum(self.penalty_structure == UNPENALIZED)]
        retry_counter = 0

        all_failing = np.zeros(grad_solution.shape, np.bool)

        for lagrange_new, lagrange_cur in zip(lseq[1:], lseq[:-1]):
            self.lagrange = lagrange_new
            tol = inner_tol
            active_old = self.active.copy()
            num_tries = 0
            debug = False
            coef_stop = True
            while True:
                strong, strong_selector = self.strong_set(lagrange_cur, 
                                                          lagrange_new, grad=grad_solution)

                subproblem_set = self.ever_active + all_failing
                self.final_inv_step, grad, sub_soln, penalty_structure \
                    = self.solve_subproblem(subproblem_set,
                                            lagrange_new,
                                            tol=tol,
                                            start_inv_step=self.final_inv_step,
                                            debug=debug,
                                            coef_stop=coef_stop)

                p = self.shape[1]

                self.solution[subproblem_set][:] = sub_soln
                # this only corrects the gradient on the subproblem_set
                grad_solution[subproblem_set][:] = grad

                strong_problem = self.restricted_problem(strong, lagrange_new)[0]
                strong_soln = self.solution[strong]
                strong_grad = (strong_problem.smooth_objective(strong_soln, mode='grad') + 
                               self.elastic_net[strong].objective(strong_soln, mode='grad'))
                strong_penalty_structure = self.penalty_structure[strong]

                strong_failing = check_KKT(strong_grad, strong_soln, lagrange_new, 
                                           strong_penalty_structure, debug=False)
                if np.any(strong_failing):
                    all_failing += strong_selector.adjoint_map(strong_failing)
                else:
                    self.solution[subproblem_set][:] = sub_soln
                    grad_solution = self.grad()
                    all_failing = check_KKT(grad_solution, self.solution, lagrange_new, 
                                        self.penalty_structure)
                    if not all_failing.sum():
                        self.ever_active += self.solution != 0
                        break
                    else:
                        print 'failing:', np.nonzero(all_failing)[0]
                        retry_counter += 1
                        self.ever_active += all_failing

                tol /= 2.
                num_tries += 1
                if num_tries % 5 == 0:

                    self.solution[subproblem_set][:] = sub_soln
                    self.solution[~subproblem_set][:] = 0
                    grad_solution = self.grad()

                    debug = True
                    tol = inner_tol
                    #stop

            rescaled_solution = self.nonzero.adjoint_map(self.solution)
            rescaled_solutions = scipy.sparse.vstack([rescaled_solutions, rescaled_solution])
            objective.append(self.loss.smooth_objective(self.solution, mode='func'))
            dfs.append(self.ever_active.shape[0])
            gc.collect()

            print lagrange_cur / self.lagrange_max, lagrange_new, (self.solution != 0).sum(), 1. - objective[-1] / objective[0], list(self.lagrange_sequence).index(lagrange_new), np.fabs(rescaled_solution).sum()

        objective = np.array(objective)
        output = {'devratio': 1 - objective / objective.max(),
                  'df': dfs,
                  'lagrange': self.lagrange_sequence,
                  'scalings': scalings,
                  'beta':rescaled_solutions.T}

        return output

    # Some common loss factories

    @classmethod
    def logistic(cls, X, Y, *args, **keyword_args):
        return cls(logistic_factory(Y), X, *args, **keyword_args)

    @classmethod
    def squared_error(cls, X, Y, *args, **keyword_args):
        return cls(squared_error_factory(Y), X, *args, **keyword_args)

class loss_factory(object):

    def __init__(self, response):
        self._response = np.asarray(response)

    def __call__(self, X):
        raise NotImplementedError

    def get_response(self):
        return self._response

    def set_response(self, response):
        self._response = response
    response = property(get_response, set_response)

class logistic_factory(loss_factory):

    def __call__(self, X):
        return logistic_loss(X, self.response, coef=0.5)

class squared_error_factory(loss_factory):

    def __call__(self, X):
        n = self.response.shape[0]
        return squared_error(X, -self.response, coef=1./n)


class nesta(lasso):

    # atom_factory takes candidate_set, epsilon

    def __init__(self, loss_factory, X, atom_factory, epsilon=None,
                 **lasso_keywords):
        self.atom = atom 
        self.epsilon_values = epsilon
        lasso.__init__(self, loss_factory, X, **lasso_keywords)

    @property
    def problem(self):
        if not hasattr(self, '_problem'):
            self.epsilon = self.epsilon_values[0]
        return self._problem

    def set_epsilon(self, epsilon):
        self._epsilon = epsilon
        if not hasattr(self, 'dual_term'):
            self.nesta_term = self.atom.smoothed(iq(epsilon, 0, 0, 0))
            self.nesta_term.smooth_objective(np.zeros(self.shape[1]), mode='grad')
            self.dual_term = self.nesta_term.grad
        else:
            self.nesta_term = self.atom.smoothed(iq(epsilon, self.dual_term, 0, 0))
        subproblem = self.restricted_problem(np.ones(self.shape[1], np.bool), self.lagrange_max)[0]
        nesta_term = self.nesta_term
        nesta_smooth = smooth_sum([subproblem.smooth_atom, nesta_term])
        self._problem = simple_problem(nesta_smooth, subproblem.proximal_atom)

    def get_epsilon(self):
        return self._epsilon
    epsilon = property(get_epsilon, set_epsilon)

    def set_final_inv_step(self):
        if not hasattr(self, "_final_inv_step_lookup"):
            self._final_inv_step_lookup = {}
        self._final_inv_step_lookup[self.epsilon]

    def get_final_inv_step(self):
        if not hasattr(self, "_final_inv_step_lookup"):
            self._final_inv_step_lookup = {}
        if self.epsilon not in self._final_inv_step_lookup:
            self._final_inv_step_lookup[self.epsilon] = \
                max(self._final_inv_step_lookup.value())
        return self._final_inv_step_lookup[self.epsilon]
    final_inv_step = property(get_final_inv_step, set_final_inv_step)

    def form_nesta_term(self, epsilon, candidate_set):
        pass

    def solve_subproblem(self, candidate_set, lagrange_new, **solve_args):
    
        # try to solve the problem with the active set
        for epsilon in self.epsilon_values:
            print 'epsilon:', epsilon
            self.epsilon = epsilon
            subproblem, selector, penalty_structure = self.restricted_problem(candidate_set, lagrange_new)
            nesta_term = affine_smooth(self.nesta_term, adjoint(selector), store_grad=True)
            nesta_smooth = smooth_sum([subproblem.smooth_atom, nesta_term])
            nesta_problem = simple_problem(nesta_smooth, subproblem.proximal_atom)

            nesta_problem.coefs[:] = selector.linear_map(self.solution)
            sub_soln = nesta_problem.solve(**solve_args)
            self.solution[:] = selector.adjoint_map(sub_soln)
            self.dual_term[:] = self.nesta_term.grad
            print 'DUALDUAL: ', self.dual_term
            self.final_inv_step = nesta_problem.final_inv_step
        grad = nesta_problem.smooth_objective(sub_soln, mode='grad') 
        
        return self.final_inv_step, grad, sub_soln, penalty_structure


def check_KKT(grad, solution, lagrange, penalty_structure, subset=None, tol=1.0e-02,
              debug=False):
    '''
    Verify that the KKT conditions for the LASSO possibly with unpenalized coefficients
    is satisfied for (grad, solution) where grad is the gradient of the loss evaluated
    at solution.

    Does not check unpenalized coefficients as solver has should have returned
    a solution at which the gradient is 0 on these coordinates.

    '''

    failing = np.zeros(grad.shape)

    if subset is None:
        subset = slice(None, None, None)

    # L1 check

    # Check subgradient is feasible
    l1_set = (penalty_structure == L1_PENALTY)[subset]
    if l1_set.sum():

        g_l1 = grad[l1_set]
        failing[l1_set] += np.fabs(g_l1) > lagrange * (1 + tol)
        if debug:
            print 'l1 (dual) feasibility:', np.fabs(g_l1), lagrange * (1 + tol)

        # Check that active coefficients are on the boundary 
        soln_l1 = solution[l1_set]
        active_l1 = soln_l1 != 0

        failing_l1 = np.zeros(g_l1.shape)
        failing_l1[active_l1] += np.fabs(-g_l1[active_l1] / lagrange - np.sign(soln_l1[active_l1])) >= tol 
        failing[l1_set] += failing_l1

        if debug:
            print 'l1 (dual) tightness:', np.fabs(-g_l1[active_l1] / lagrange - np.sign(soln_l1[active_l1]))

    # Positive part

    # Check subgradient is feasible
    pp_set = (penalty_structure == POSITIVE_PART)[subset]
    if pp_set.sum():
        g_pp = -grad[pp_set]
        failing_pp = np.zeros(g_pp.shape)
        failing[pp_set] += g_pp > lagrange * (1 + tol)
        if debug:
            print 'positive part (dual) feasibility:', g_pp > lagrange * (1 + tol)

        # Check that active coefficients are on the boundary 
        soln_pp = solution[pp_set]
        active_pp = soln_pp != 0

        failing_pp[active_pp] += g_pp[active_pp] / lagrange - 1 >= tol 
        if debug:
            print 'positive part (dual) tightness:', g_pp[active_pp] / lagrange - 1
        failing[pp_set] += failing_pp

    return failing > 0


def newsgroup():
    import scipy.io

    D = scipy.io.loadmat('newsgroup.mat')
    X = D['X']; Y = D['Y']

    newsgroup_lasso = lasso.logistic(X, Y)
    newsgroup_lasso.main()
