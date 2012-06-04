import numpy as np
import scipy.sparse

from .affine import power_L, normalize, selector, identity
from .atoms import l1norm 
from .smooth import logistic_loss
from .quadratic import squared_error
from .separable import separable_problem
from .simple import simple_problem
from .identity_quadratic import identity_quadratic as iq

import gc
class lasso(object):

    def __init__(self, loss_factory, X, elastic_net=iq(0,0,0,0),
                 alpha=0., intercept=True,
                 lagrange_proportion = 0.05,
                 nstep = 100,
                 scale=True,
                 center=True):
        self.loss_factory = loss_factory

        # the normalization of X
        self.intercept = intercept
        if self.intercept:
            if scipy.sparse.issparse(X):
                self._X1 = scipy.sparse.hstack([np.ones((X.shape[0], 1)), X]).tocsc() 
            else:
                self._X1 = np.hstack([np.ones((X.shape[0], 1)), X])
            self._Xn = normalize(self._X1, center=center, scale=scale, intercept_column=0)
        else:
            self._Xn = normalize(X, center=center, scale=scale)

        # the penalty parameters
        self.alpha = alpha
        self.lagrange_proportion = lagrange_proportion
        self.nstep = nstep
        self._elastic_net = elastic_net.collapsed()


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
        # what is X changes?
        if not hasattr(self, '_loss'):
            self._loss = self.loss_factory(self._Xn)
        return self._loss

    @property
    def null_solution(self):
        if not hasattr(self, "_null_soln"):
            n, p = self.Xn.dual_shape[0], self.Xn.primal_shape[0]
            self._null_soln = np.zeros(p)
            if self.intercept:
                null_design = np.ones((n,1))
                null_loss = self.loss_factory(null_design)
                self._null_soln[0] = null_loss.solve()
        return self._null_soln

    @property
    def lagrange_max(self):
        if not hasattr(self, "_lagrange_max"):
            null_soln = self.null_solution
            if self.intercept:
                self._lagrange_max = np.fabs(self.loss.smooth_objective(null_soln, 'grad'))[1:].max()
            else:
                self._lagrange_max = np.fabs(self.loss.smooth_objective(null_soln, 'grad')).max()
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
        p = self.Xn.primal_shape[0]
        if not hasattr(self, "_problem"):
            if self.intercept:
                linear_slice = slice(1, p)
                linear_penalty = l1norm(p-1, lagrange=self.lagrange_max)
                self._problem = separable_problem(self.loss, self.Xn.primal_shape, [linear_penalty], [linear_slice])
                self._problem.coefs[:] = self.null_solution
            else:
                penalty = l1norm(p, lagrange=self.lagrange_max)
                self._problem = simple_problem(self.loss, penalty)
        return self._problem

    def get_lagrange(self):
        if self.intercept:
            return self._problem.nonsmooth_atom.atoms[0].lagrange
        else:
            return self._problem.nonsmooth_atom.lagrange

    def set_lagrange(self, lagrange):
        if self.intercept:
            self._problem.nonsmooth_atom.atoms[0].lagrange = lagrange
            self._problem.nonsmooth_atom.atoms[0].quadratic = self.elastic_net
        else:
            self._problem.nonsmooth_atom.lagrange = lagrange
            self._problem.nonsmooth_atom.quadratic = self.elastic_net
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

    @property
    def penalized(self):
        if not hasattr(self, '_penalized'):
            p = self.Xn.primal_shape[0]
            if self.intercept:
                self._penalized = selector(slice(1,p), (p,))
            else:
                self._penalized = identity(p)
        return self._penalized

    def grad(self):
        '''
        Gradient at current value. This includes the gradient
        of the smooth loss as well as the gradient of the elastic net part.
        This is used for determining whether the KKT conditions are met
        and which coefficients are in the strong set.
        '''
        gsmooth = self.loss.smooth_objective(self.solution, 'grad')
        p = self.penalized
        gquad = self.elastic_net.objective(p.linear_map(self.solution), 'grad')

        return gsmooth + p.adjoint_map(gquad)

    def strong_set(self, lagrange_cur, lagrange_new, 
                   slope_estimate=1, grad=None):
        if grad is None:
            grad = self.grad()
        s = self.penalized
        value = np.zeros(grad.shape, np.bool)
        value += s.adjoint_map(np.fabs(s.linear_map(grad)) < (slope_estimate+1) * lagrange_new - slope_estimate*lagrange_cur)
        return ~value

    def restricted_problem(self, candidate_set, lagrange):
        '''
        Assumes the candidate set includes intercept as first column.
        '''
        Xslice = self.Xn.slice_columns(candidate_set)
        loss = self.loss_factory(Xslice)
        if self.intercept:
            Xslice.intercept_column = 0
            linear_slice = slice(1, Xslice.primal_shape[0])
            linear_penalty = l1norm(Xslice.primal_shape[0]-1, lagrange=lagrange)
            problem_sliced = separable_problem(loss, Xslice.primal_shape, [linear_penalty], [linear_slice])
        else:
            penalty = l1norm(Xslice.primal_shape[0], lagrange=lagrange)
            problem_sliced = simple_problem(loss, penalty)
        candidate_selector = selector(candidate_set, self.Xn.primal_shape)
        return problem_sliced, candidate_selector

    def check_KKT(self, tol=1.0e-02):
        '''
        Verify that the KKT conditions for the LASSO possibly with unpenalized coefficients
        is satisfied for (grad, solution) where grad is the gradient of the loss evaluated
        at solution.
        '''
        grad = self.grad()
        solution = self.solution
        s = self.penalized
        lagrange = self.lagrange

        soln_s = s.linear_map(solution)
        g_s = s.linear_map(grad)
        failing_s = np.zeros(g_s.shape)
        failing = np.zeros(grad.shape)

        # Check all coefficients
        failing += s.adjoint_map(np.fabs(g_s) > lagrange * (1 + tol))

        # Check the active coefficients
        active = soln_s != 0
        failing_s[active] += np.fabs(g_s[active] / lagrange + np.sign(soln_s[active])) >= tol 
        failing += s.adjoint_map(failing_s)

        return failing

    def solve_subproblem(self, lagrange_new, **solve_args):
    
        # try to solve the problem with the active set
        subproblem, selector = self.restricted_problem(self.strong, lagrange_new)
        subproblem.coefs[:] = selector.linear_map(self.solution)
        sub_soln = subproblem.solve(**solve_args)
        self.solution[:] = selector.adjoint_map(sub_soln)

        final_inv_step = subproblem.final_inv_step
        return final_inv_step

    def main(self, inner_tol=1.e-5):

        # scaling will be needed to get coefficients on original scale   
        if self.Xn.scale:
            scalings = np.asarray(self.Xn.col_stds).reshape(-1)
        else:
            scalings = np.ones(self.Xn.primal_shape)

        # take a guess at the inverse step size
        final_inv_step = self.lipschitz / 1000
        lseq = self.lagrange_sequence

        # first solution corresponding to all zeros except intercept 

        self.solution[:] = self.null_solution

        self.strong = self.strong_set(lseq[0], lseq[1])
        grad_solution = self.grad().copy()

        p = self.Xn.primal_shape[0]

        rescaled_solutions = scipy.sparse.csr_matrix(self.solution / scalings)

        objective = [self.loss.smooth_objective(self.solution, 'func')]
        dfs = [1]
        retry_counter = 0


        for lagrange_new, lagrange_cur in zip(lseq[1:], lseq[:-1]):
            self.lagrange = lagrange_new
            tol = inner_tol
            active_old = self.active.copy()
            num_tries = 0
            debug = False
            coef_stop = True
            while True:
                self.strong = self.strong_set(lagrange_cur, lagrange_new, grad=grad_solution)
                final_inv_step = self.solve_subproblem(lagrange_new,
                                                       tol=tol,
                                                       start_inv_step=final_inv_step,
                                                       debug=debug,
                                                       coef_stop=coef_stop)
                active = self.active
                if active_old.sum() <= active.sum() and (~active_old * active).sum() == 0:
                    failing = self.check_KKT()
                    if not failing.sum():
                        grad_solution = self.grad().copy()
                        break
                    else:
                        retry_counter += 1
                        active += self.strong
                else:
                    self.strong += active
                    failing = self.check_KKT()
                    if not failing.sum():
                        grad_solution = self.grad().copy()
                        break

                tol /= 2.
                num_tries += 1
                if num_tries % 5 == 0:
                    debug=True
                    tol = inner_tol

            rescaled_solution = self.solution / scalings
            rescaled_solutions = scipy.sparse.vstack([rescaled_solutions, rescaled_solution])
            objective.append(self.loss.smooth_objective(self.solution, mode='func'))
            dfs.append(active.shape[0])
            gc.collect()

            print lagrange_cur / self.lagrange_max, lagrange_new, (self.solution != 0).sum(), 1. - objective[-1] / objective[0], list(self.lagrange_sequence).index(lagrange_new), np.fabs(rescaled_solution[-1]).sum()

        objective = np.array(objective)
        output = {'devratio': 1 - objective / objective.max(),
                  'df': dfs,
                  'lagrange': self.lagrange_sequence,
                  'scalings': scalings,
                  'beta':rescaled_solutions}

        return output

    @staticmethod
    def logistic(X, Y, **keyword_args):
        def logistic_factory(X):
            return logistic_loss(X, Y, coef=0.5)
        return lasso(logistic_factory, X, **keyword_args)

    @staticmethod
    def squared_error(X, Y, **keyword_args):
        n = Y.shape[0]
        def squared_error_factory(X):
            return squared_error(X, Y, coef=1./n)
        return lasso(squared_error_factory, X, **keyword_args)

def newsgroup():
    import scipy.io

    D = scipy.io.loadmat('newsgroup.mat')
    X = D['X']; Y = D['Y']

    newsgroup_lasso = lasso.logistic(X, Y)
    newsgroup_lasso.main()
