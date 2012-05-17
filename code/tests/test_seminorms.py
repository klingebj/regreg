import numpy as np
import regreg.atoms as A
import regreg.api as rr
import nose.tools as nt

def ac(x,y, msg=None):
    v = np.linalg.norm(x-y) <= 1.0e-04 * max([1, np.linalg.norm(x), np.linalg.norm(y)])
    if not v:
        print 'msg: ', msg
    nt.assert_true(v)

def test_proximal_maps():
    bound = 0.14
    lagrange = 0.13
    shape = 20

    Z = np.random.standard_normal(shape)
    for L in [0.5,1,0.1]:
        for primal, dual in sorted(A.conjugate_seminorm_pairs.items()):
            p = primal(shape, lagrange=lagrange)
            d = p.conjugate
            yield nt.assert_equal, d, dual(shape, bound=lagrange)
            yield ac, Z-p.proximal(1, Z, 0), d.proximal(1, Z, 0), 'testing dual of projections starting from primal %s, %s' % (primal, dual)
            yield ac, p.lagrange_prox(Z, lipschitz=L), Z-d.bound_prox(Z*L, lipschitz=1./L)/L, 'testing lagrange_prox and bound_prox starting from primal %s, %s' % (primal, dual)

            # some arguments of the constructor

            nt.assert_raises(AttributeError, setattr, p, 'bound', 4.)
            nt.assert_raises(AttributeError, setattr, d, 'lagrange', 4.)

            nt.assert_raises(AttributeError, setattr, p, 'bound', 4.)
            nt.assert_raises(AttributeError, setattr, d, 'lagrange', 4.)

            p2 = primal(shape, lagrange=lagrange)
            p2.set_quadratic(L, Z, 0, 0)
            problem = rr.simple_problem.nonsmooth(p2)
            solver = rr.FISTA(problem)
            solver.fit(tol=1.0e-12)

            yield ac, p.proximal(L, Z, 0), solver.composite.coefs, 'solving prox with simple_problem.nonsmooth %s, %s' % (primal, dual)

            loss = rr.quadratic.shift(-Z, coef=0.5*L)
            problem = rr.simple_problem(loss, p)
            solver = rr.FISTA(problem)

            # restarting is acting funny
            solver.fit(tol=1.0e-12, min_its=100)

            yield ac, p.proximal(L, Z, 0), solver.composite.coefs, 'solving prox with simple_problem %s, %s' % (primal, dual)

            loss = rr.quadratic.shift(-Z, coef=0.5*L)
            problem = rr.simple_problem(loss, p)
            solver = rr.FISTA(problem)

            solver.fit(tol=1.0e-12, monotonicity_restart=False, min_its=100)

            yield ac, p.proximal(L, Z, 0), solver.composite.coefs, 'solving prox with simple_problem %s, %s no monotonicity_restart' % (primal, dual)

            loss = rr.quadratic.shift(-Z, coef=0.5*L)
            problem = rr.separable_problem.singleton(p, loss)
            solver = rr.FISTA(problem)
            solver.fit(tol=1.0e-12)

            yield ac, p.proximal(L, Z, 0), solver.composite.coefs, 'solving primal prox with separable_atom.singleton %s, %s' % (primal, dual)

            loss = rr.quadratic.shift(-Z, coef=0.5*L)
            problem = rr.container(loss, p)
            solver = rr.FISTA(problem)
            solver.fit(tol=1.0e-12)

            yield ac, p.proximal(L, Z, 0), solver.composite.coefs, 'solving primal prox with container %s, %s' % (primal, dual)

            loss = rr.quadratic.shift(-Z, coef=0.5*L)
            problem = rr.simple_problem(loss, d)
            solver = rr.FISTA(problem)
            solver.fit(tol=1.0e-12, monotonicity_restart=False)
            yield ac, d.proximal(L, Z, 0), solver.composite.coefs, 'solving dual prox with simple_problem no monotonocity %s, %s' % (primal, dual)

            problem = rr.container(d, loss)
            solver = rr.FISTA(problem)
            solver.fit(tol=1.0e-12)
            yield ac, d.proximal(L, Z, 0), solver.composite.coefs, 'solving dual prox with container %s, %s' % (primal, dual)
            
            loss = rr.quadratic.shift(-Z, coef=0.5*L)
            problem = rr.separable_problem.singleton(d, loss)
            solver = rr.FISTA(problem)
            solver.fit(tol=1.0e-12)

            yield ac, d.proximal(L, Z, 0), solver.composite.coefs, 'solving primal prox with separable_atom.singleton %s, %s' % (primal, dual)


            d = dual(shape, bound=bound)
            p = d.conjugate
            yield nt.assert_equal, p, primal(shape, lagrange=bound)
            yield ac, Z-p.proximal(L, Z, 0), d.proximal(1./L, Z*L, 0)/L, 'using different lipschitz constant with proximal starting from dual %s, %s' % (primal, dual)
            yield ac, p.lagrange_prox(Z, L), Z-d.bound_prox(Z*L,1./L)/L, 'using different lipschitz constant with lagrange_prox starting from dual, bound_prox %s, %s' % (primal, dual)

    #        yield ac, d.proximal_optimum(Z)[1], p.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2

def test_linear_term_proximal():
    bound = 0.14
    lagrange = 0.5
    shape = 20

    Z = np.random.standard_normal(shape)
    W = 0.02 * np.random.standard_normal(shape)
    linq = rr.identity_quadratic(0,0,W,0)

    for primal, dual in A.conjugate_seminorm_pairs.items():
        for L in [0.5,1]:
            p = primal(shape, lagrange=lagrange, quadratic=linq)
            d = p.conjugate
            yield nt.assert_equal, d, dual(shape, bound=lagrange)
            yield ac, p.proximal(L, Z, 0), Z-d.proximal(1./L, Z*L, 0)/L, 'using different lipschitz constant with proximal starting form primal %s, %s' % (primal, dual)
            yield ac, p.lagrange_prox(Z,L), Z-d.bound_prox(Z*L,1./L)/L, 'using different lipschitz constant with lagrange_prox, bound_prox starting form primal %s, %s' % (primal, dual)

            d = dual(shape, bound=bound, quadratic=linq)
            p = d.conjugate
            yield nt.assert_equal, p, primal(shape, lagrange=bound)
            yield ac, p.proximal(L, Z, 0), Z-d.proximal(1./L, Z*L, 0)/L, 'using different lipschitz constant with proximal starting form dual %s, %s' % (primal, dual)
            yield ac, p.lagrange_prox(Z,L), Z-d.bound_prox(Z*L,1./L)/L, 'using different lipschitz constant with lagrange_prox, bound_prox starting form dual %s, %s' % (primal, dual)
            
            ##        yield ac, d.proximal_optimum(Z)[1], p.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2

            loss = rr.quadratic.shift(-Z, coef=0.5*L)
            problem = rr.separable_problem.singleton(p, loss)
            solver = rr.FISTA(problem)
            solver.fit(tol=1.0e-12)

            yield ac, p.proximal(L, Z, 0), solver.composite.coefs, 'singleton primal prox %s %s ' % (primal, dual)
            problem = rr.separable_problem.singleton(d, loss)

            solver = rr.FISTA(problem)
            solver.fit(tol=1.0e-12)
            yield ac, d.proximal(L, Z, 0), solver.composite.coefs, 'singleton dual prox %s %s ' % (primal, dual)

            p2 = primal(shape, lagrange=lagrange, quadratic=linq)
            p2.set_quadratic(L, Z, 0, 0)
            problem = rr.simple_problem.nonsmooth(p2)
            solver = rr.FISTA(problem)
            solver.fit(tol=1.0e-12)

            yield ac, p.proximal(L, Z, 0), solver.composite.coefs, 'solving prox with simple_problem.nonsmooth %s, %s' % (primal, dual)

            loss = rr.quadratic.shift(-Z, coef=0.5*L)
            problem = rr.simple_problem(loss, p)
            solver = rr.FISTA(problem)

            # restarting is acting funny
            solver.fit(tol=1.0e-12, min_its=100)

            yield ac, p.proximal(L, Z, 0), solver.composite.coefs, 'solving prox with simple_problem with monotonicity %s, %s' % (primal, dual)

            loss = rr.quadratic.shift(-Z, coef=0.5*L)
            problem = rr.simple_problem(loss, p)
            solver = rr.FISTA(problem)

            solver.fit(tol=1.0e-12, monotonicity_restart=False, min_its=100)

            yield ac, p.proximal(L, Z, 0), solver.composite.coefs, 'solving prox with simple_problem %s, %s no monotonicity_restart' % (primal, dual)

            loss = rr.quadratic.shift(-Z, coef=0.5*L)
            problem = rr.separable_problem.singleton(p, loss)
            solver = rr.FISTA(problem)
            solver.fit(tol=1.0e-12)

            yield ac, p.proximal(L, Z, 0), solver.composite.coefs, 'solving primal prox with separable_atom.singleton %s, %s' % (primal, dual)

            loss = rr.quadratic.shift(-Z, coef=0.5*L)
            problem = rr.container(loss, p)
            solver = rr.FISTA(problem)
            solver.fit(tol=1.0e-12)

            yield ac, p.proximal(L, Z, 0), solver.composite.coefs, 'solving primal prox with container %s, %s' % (primal, dual)

            loss = rr.quadratic.shift(-Z, coef=0.5*L)
            problem = rr.simple_problem(loss, d)
            solver = rr.FISTA(problem)
            solver.fit(tol=1.0e-12, monotonicity_restart=False)
            yield ac, d.proximal(L, Z, 0), solver.composite.coefs, 'solving dual prox with simple_problem no monotonicity %s, %s' % (primal, dual)

            problem = rr.container(d, loss)
            solver = rr.FISTA(problem)
            solver.fit(tol=1.0e-12)
            yield ac, d.proximal(L, Z, 0), solver.composite.coefs, 'solving dual prox with container %s, %s' % (primal, dual)
            
            loss = rr.quadratic.shift(-Z, coef=0.5*L)
            problem = rr.separable_problem.singleton(d, loss)
            solver = rr.FISTA(problem)
            solver.fit(tol=1.0e-12)

            yield ac, d.proximal(L, Z, 0), solver.composite.coefs, 'solving primal prox with separable_atom.singleton %s, %s' % (primal, dual)

def test_offset_proximal():
    bound = 0.14
    lagrange = 0.5
    shape = 50

    L = 0.5
    Z = np.random.standard_normal(shape)
    W = 0.02 * np.random.standard_normal(shape)
    for primal, dual in A.conjugate_seminorm_pairs.items():
        p = primal(shape, lagrange=lagrange, offset=W)
        d = p.conjugate
        print p, d
        yield nt.assert_equal, d, dual(shape, bound=lagrange)
        yield ac, p.proximal(L, Z, 0), Z-d.proximal(1./L, Z*L, 0)/L, 'using different lipschitz constant with proximal starting from primal %s, %s' % (primal, dual)
        yield ac, p.lagrange_prox(Z,L), Z-d.bound_prox(Z*L,1./L)/L, 'using different lipschitz constant with lagrange_prox, bound_prox starting from primal %s, %s' % (primal, dual)

        d = dual(shape, bound=bound, offset=W)
        p = d.conjugate
        yield nt.assert_equal, p, primal(shape, lagrange=bound)
        yield ac, p.proximal(L, Z, 0), Z-d.proximal(1./L, Z*L, 0)/L, 'using different lipschitz constant with proximal starting from dual %s, %s' % (primal, dual)
        yield ac, p.lagrange_prox(Z,L), Z-d.bound_prox(Z*L,1./L)/L, 'using different lipschitz constant with lagrange_prox, bound_prox starting from dual %s, %s' % (primal, dual)
#        yield ac, d.proximal_optimum(Z)[1], p.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2

def test_offset_and_linear_term_proximal():
    bound = 0.14
    lagrange = 0.5
    shape = 50

    L = 0.5
    Z = np.random.standard_normal(shape)
    W = 0.02 * np.random.standard_normal(shape)
    U = 0.02 * np.random.standard_normal(shape)
    linq = rr.identity_quadratic(0,0,U,0)
    for primal, dual in A.conjugate_seminorm_pairs.items():
        p = primal(shape, lagrange=lagrange, offset=W, quadratic=linq)
        d = p.conjugate
        print p, d
        yield nt.assert_equal, d, dual(shape, bound=lagrange)
        yield ac, p.proximal(L, Z, 0), Z-d.proximal(1./L, Z*L, 0)/L, 'using different lipschitz constant with proximal starting from dual %s, %s' % (primal, dual)
        yield ac, p.lagrange_prox(Z,L), Z-d.bound_prox(Z*L,1./L)/L, 'using different lipschitz constant with lagrange_prox, bound_prox starting from dual %s, %s' % (primal, dual)

        d = dual(shape, bound=bound, offset=W, quadratic=linq)
        p = d.conjugate
        yield nt.assert_equal, p, primal(shape, lagrange=bound)
        yield ac, p.proximal(L, Z, 0), Z-d.proximal(1./L, Z*L, 0)/L, 'using different lipschitz constant with proximal starting from dual %s, %s' % (primal, dual)
        yield ac, p.lagrange_prox(Z,L), Z-d.bound_prox(Z*L,1./L)/L, 'using different lipschitz constant with lagrange_prox, bound_prox starting from dual %s, %s' % (primal, dual)
#        yield ac, d.proximal_optimum(Z)[1], p.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2

def test_atoms():

    nt.assert_raises(ValueError, rr.l1norm, 40)
    nt.assert_raises(ValueError, rr.l1norm, 40, bound=1, lagrange=1)
    nt.assert_raises(ValueError, rr.l1norm, 40, bound=-1)

    p1 = rr.l1norm(40, lagrange=3)
    p2 = rr.l1norm(40, lagrange=2)
    p3 = rr.l1norm(40, bound=2)
    p4 = rr.l2norm(30, bound=3)
    ps = [p1, p2, p3, p4]

    for i, p in enumerate(ps):
        for j in range(i):
            nt.assert_not_equal(p, ps[j])

    nt.assert_raises(AttributeError, setattr, p4, 'lagrange', 4)
    nt.assert_raises(AttributeError, setattr, p2, 'bound', 4)



