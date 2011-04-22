from smooth import smoothed_seminorm, affine_atom
import numpy as np

if __name__ == "__main__":
    import atoms
    hinge = atoms.positive_part(501)
    smoothed_hinge = smoothed_seminorm(hinge, epsilon=0.8)

    import pylab
    x = np.linspace(-2,2,501)
    pylab.plot(x, [smoothed_hinge.smooth_eval(xx, mode='func') for xx in x])
    smoothed_hinge = smoothed_seminorm(hinge, epsilon=0.5)
    pylab.plot(x, [smoothed_hinge.smooth_eval(xx, mode='func') for xx in x])
    smoothed_hinge = smoothed_seminorm(hinge, epsilon=1.0e-06)
    pylab.plot(x, [smoothed_hinge.smooth_eval(xx, mode='func') for xx in x])


    X = np.random.standard_normal((1000,20))
    Y = np.random.standard_normal(1000)
    smooth_hinge_loss = affine_atom(smoothed_hinge, -X, Y)
    b = np.random.standard_normal(20)
    smooth_hinge_loss.smooth_eval(b)
