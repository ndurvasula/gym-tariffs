import numpy as np
import GPy

from scipy.optimize import minimize
from scipy.stats import norm

class AI():

    def __init__(self, kv = 1, lv = 1, kl = 1, b = -1, mu, sigma, D, N, price, o=False):
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
        
        self.N = n
        self.price = price

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

        if o:
            self.GP.optimize()

    def opt(rphi):

        theta = self.GP.predict(np.array([rphi]))[0][0]
        return -1* (-rphi + self.mu + self.D*theta)

    def dopt(rphi):

        dtheta = self.GP.predictive_gradients(np.array([rphi]))[0][0][0]
        return -1* (-1 + self.D*dtheta)

    def act():

        rphi = mimimize(opt, 0, method='BFGS', jac=dopt).x
        phi = []
        theta = self.GP.predict(np.array([rphi]))[0][0]
        psi = rphi/(self.D*theta + np.random.normal(self.mu, self.sigma))
        
        if psi > 1:
            psi = 1
            
        if psi < 0:
            psi = 0

        psi /= 2
        
        K = .3
        MEANS = np.array([i*1.0/(self.N-1) for i in range(self.N)])
        STD = 1/(6*(self.N-1))
        
        pmf = np.array([norm.cdf(psi+.5+K, loc=MEANS[i], scale=STD) - norm.cdf(psi+.5-K, loc=MEANS[i], scale=STD) for i in range(self.N)])
        pmf /= np.sum(pmf)

        price = 0

        while rphi > 0:
            product = list(np.random.multinomial(1, pmf)).index(1)
            rphi -= self.price(product)
            price += self.price(product)
            phi.append([product])

        return np.array(phi), price

#Arbitary as of now
def price(product):
    return product**2

TIME = 0
NATIONS = 3
N = 3
D = [5000000000,1000000000,2000000000,3000000000] #untariffed profit in N_0

#Mu_i is the sum of the profits from all nations (not N_0) - sum of other nations' profits in N_i
MU = [2000000000,1000000000,4500000000]
STD = [5000000,6000000,3000000]

KV = [1,.5,5]


            
            
            
        
        
        
        
    
