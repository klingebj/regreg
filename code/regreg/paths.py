import numpy as np
import scipy.sparse

from .affine import power_L, normalize, selector
from .atoms import l1norm 
from .smooth import logistic_loss
from .quadratic import squared_error
from .separable import separable_problem

import gc
class lasso(object):

    intercept = True
    lmin = 0.05
    nstep = 100

    def __init__(self, loss_factory, X, lmin=0.05):
        self.loss_factory = loss_factory
        self._X1 = scipy.sparse.hstack([np.ones((X.shape[0], 1)), X]).tocsc() 
        self._Xn = normalize(self._X1, center=True, scale=True, intercept_column=0)

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
            null_design = np.ones((n,1))
            null_loss = self.loss_factory(null_design)
            value = np.zeros(p)
            value[0] = null_loss.solve()
            self._null_soln = value
        return self._null_soln

    @property
    def lagrange_max(self):
        if not hasattr(self, "_lagrange_max"):
            null_soln = self.null_solution
            self._lagrange_max = np.fabs(self.loss.smooth_objective(null_soln, 'grad'))[1:].max()
        return self._lagrange_max

    @property
    def lagrange_sequence(self):
        return self.lagrange_max * np.exp(np.linspace(np.log(0.05), 0, 100))[::-1]

    @property
    def problem(self):
        p = self.Xn.primal_shape[0]
        if not hasattr(self, "_problem"):
            linear_slice = slice(1, p)
            linear_penalty = l1norm(p-1, lagrange=self.lagrange_max)
            self._problem = separable_problem(self.loss, self.Xn.primal_shape, [linear_penalty], [linear_slice])
            self._problem.coefs[:] = self.null_solution
        return self._problem

    def get_lagrange(self):
        return self._problem.nonsmooth_atom.atoms[0].lagrange

    def set_lagrange(self, lagrange):
        self._problem.nonsmooth_atom.atoms[0].lagrange = lagrange
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
            self._penalized = selector(slice(1,p), (p,))
        return self._penalized

    def grad(self):
        '''
        Gradient at current value.
        '''
        return self.loss.smooth_objective(self.solution, 'grad')

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
        if self.intercept:
            Xslice.intercept_column = 0
        loss = self.loss_factory(Xslice)
        linear_slice = slice(1, Xslice.primal_shape[0])
        linear_penalty = l1norm(Xslice.primal_shape[0]-1, lagrange=lagrange)
        candidate_selector = selector(candidate_set, self.Xn.primal_shape)
        problem_sliced = separable_problem(loss, Xslice.primal_shape, [linear_penalty], [linear_slice])
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

    def main(self):

        # scaling will be needed to get coefficients on original scale   
        scalings = np.asarray(self.Xn.col_stds).reshape(-1)

        # take a guess at the inverse step size
        final_inv_step = self.lipschitz / 1000
        lseq = self.lagrange_sequence

        # first solution corresponding to all zeros except intercept 

        self.solution[:] = self.null_solution

        self.strong = self.strong_set(lseq[0], lseq[1])
        grad_solution = self.grad().copy()

        p = self.Xn.primal_shape[0]

        rescaled_solutions = np.zeros((self.nstep, p-1))
        rescaled_solutions[0] = self.solution[1:]

        objective = [self.loss.smooth_objective(self.solution, 'func')]
        dfs = [1]
        retry_counter = 0

        idx = 1
        for lagrange_new, lagrange_cur in zip(lseq[1:], lseq[:-1]):
            self.lagrange = lagrange_new
            tol = 1.0e-7
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
                        print 'trying again:', retry_counter, 'failing:', np.nonzero(failing)[0], active.sum()
                        active += strong
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
                    tol = 1.0e-5

            rescaled_solutions[idx] = self.solution[1:] / scalings[1:]
            objective.append(self.loss.smooth_objective(self.solution, mode='func'))
            dfs.append(active.shape[0])
            print lagrange_cur / self.lagrange_max, lagrange_new, (self.solution != 0).sum(), 1. - objective[-1] / objective[0], list(self.lagrange_sequence).index(lagrange_new), np.fabs(rescaled_solutions[idx]).sum()
            idx += 1
            gc.collect()

        objective = np.array(objective)
        output = {'devratio': 1 - objective / objective.max(),
                  'df': dfs,
                  'lagrange': lagrange_sequence,
                  'scalings': scalings,
                  'rescaled_beta': rescaled_solutions}

        scipy.io.savemat('newsgroup_results.mat', output)

    @staticmethod
    def logistic(X, Y):
        def logistic_factory(X):
            return logistic_loss(X, Y, coef=0.5)
        return lasso(logistic_factory, X)

    @staticmethod
    def squared_error(X, Y):
        n = Y.shape[0]
        def squared_error_factory(X):
            return squared_error(X, Y, coef=1./n)
        return lasso(squared_error_factory, X)

def newsgroup():
    import scipy.io

    D = scipy.io.loadmat('newsgroup.mat')
    X = D['X']; Y = D['Y']

    newsgroup_lasso = lasso.logistic(X, Y)
    newsgroup_lasso.main()
