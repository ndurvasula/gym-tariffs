import numpy as np
import GPy

from scipy.optimize import minimize

class AI():

    def __init__(self, kv = 1, lv = 1, kl = 1, b = -1, mu, sigma, D):
        #kv = kernel variance, lv = likelihood variance, kl = kernel lengthscale
        #b is bin size (max number of points the AI considers)

        self.GP = None

        self.X = np.array([])
        self.Y = np.array([])
        
        self.kv = kv
        self.lv = lv
        self.kl = kl

        self.b = b

        self.mu = mu
        self.sigma = sigma

        self.D = D

    def update(self, rphi, theta):

        if b > 0 and len(self.X) == b:
            del self.X[0]
            del self.Y[0]
            
        self.X = np.append(self.X, [rphi], axis=0)
        self.Y = np.append(self.Y, [theta], axis=0)

        self.GP = GPy.models.GPRegression(self.X, self.Y)
        
        self.GP.kern.variance = kv
        self.GP.kern.lengthscale = kl
        self.GP.likelihood.variance = lv

    def opt(rphi):

        theta = self.GP.predict(np.array([rphi]))[0][0]
        return -1* (-rphi + self.mu + self.D*theta)

    def dopt(rphi):

        dtheta = self.GP.predictive_gradients(np.array([rphi]))[0][0][0]
        return -1* (-1 + self.D*dtheta)

    def act():

        rphi = mimimize(opt, 0, method='BFGS', jac=dopt).x
        
    
