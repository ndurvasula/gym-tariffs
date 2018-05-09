import numpy as np
import GPy

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from scipy.stats import norm
from scipy.optimize import minimize

class AI():

    def __init__(self, K = .25, kv = 1, kl = 1, b = -1, mu=1, sigma=1, D=1, N=1, price=None, o=False):
        #kv = kernel variance, kl = kernel lengthscale
        #K is the spread factor over price distribution
        #b is bin size (max number of points the AI considers)

        self.GP = None

        self.X = np.array([])
        self.Y = np.array([])
        
        self.kv = kv
        self.kl = kl

        self.b = b

        self.mu = mu
        self.sigma = sigma

        self.D = D

        self.K = K
        
        self.N = n
        self.price = price

        self.action = None

    def update(self, theta):

        if b > 0 and len(self.X) == b:
            del self.X[0]
            del self.Y[0]
            
        self.X = np.append(self.X, [self.action], axis=0)
        self.Y = np.append(self.Y, [theta], axis=0)

        self.GP = GPy.models.GPRegression(self.X, self.Y)
        
        self.GP.kern.variance = kv
        self.GP.kern.lengthscale = kl

        if o:
            self.GP.optimize()

    def opt(rphi):

        theta = self.GP.predict(np.array([rphi]))[0][0]
        return -1* (-rphi + self.mu + self.D*theta)

    def dopt(rphi):

        dtheta = self.GP.predictive_gradients(np.array([rphi]))[0][0][0]
        return -1* (-1 + self.D*dtheta)

    def act(self):

        rphi = mimimize(opt, 0, method='BFGS', jac=dopt).x
        self.action = rphi
        
        theta = self.GP.predict(np.array([rphi]))[0][0]
        psi = rphi/(self.D*theta + np.random.normal(self.mu, self.sigma))
        
        if psi > 1:
            psi = 1
            
        if psi < 0:
            psi = 0

        psi /= 2
        psi = (.5-psi)
        
        
        MEANS = np.array([i*1.0/(self.N-1) for i in range(self.N)])
        STD = 1/(6*(self.N-1))
        
        pmf = np.array([norm.cdf(psi+.5+self.K, loc=MEANS[i], scale=STD) - norm.cdf(psi+.5-self.K, loc=MEANS[i], scale=STD) for i in range(self.N)])
        pmf /= np.sum(pmf)

        price = 0

        phi = np.zeros(self.N)

        while rphi > 0:
            product = list(np.random.multinomial(1, pmf)).index(1)
            rphi -= self.price(product)
            price += self.price(product)
            phi[product] += 1

        return phi, price

#Arbitary as of now
def price(product):
    return PRODUCT_BASE*(product+1)**2 #product is 0 indexed


NATIONS = 3
DAYS = 365
N = 3
PRODUCT_BASE = 100000
D = [5000000000,1000000000,2000000000,3000000000] #untariffed profit in N_0

#Mu_i is the sum of the profits from all nations (not N_0) - sum of other nations' profits in N_i
MU = [2000000000,1000000000,4500000000]
STD = [5000000,6000000,3000000]

KV = [1,.5,2]
KL = [1,.5,2]
B = [10,20,-1]
O = [False, False, True]


class TariffEnv(gym.Env):
    metadata = {'render.modes' : ['human']}
    def __init__(self):
        arr = [spaces.Discrete(int((MU[i]+D[i+1])/price(i))) for i in range(N)]
        self.observation_space = spaces.Tuple(tuple(arr))

        if DISCRETIZE:
            self.action_space = spaces.Tuple(tuple([spaces.Discrete(DELTA) for i in range(NATIONS)]))
        else:
            self.action_space= spaces.Tuple(tuple([spaces.Box(low=np.array([0]), high=np.array([1]), dtype = np.float32) for i in range(NATIONS)]))
        self.time = 0

        self.AI = [AI(kv=KV[i], kl=KL[i], o=O[i], b=B[i], N=N, D=D[i+1], price=price, mu=MU[i], sigma=STD[i]) for i in range(NATIONS)]

    def reset(self):
        self.time = 0
        arr = [0 for i in range(N)]
        self.AI = [AI(kv=KV[i], kl=KL[i], o=O[i], b=B[i], N=N, D=D[i+1], price=price, mu=MU[i], sigma=STD[i]) for i in range(NATIONS)]
        return np.array(arr)

    def step(self, action):
        self.time += 1
        
        if DISCRETIZE:
            action = action*1.0/DELTA

        for i in range(NATIONS):
            self.AI[i].update(action[i])

        res = np.array([self.AI.act() for i in range(NATIONS)])
        states, rewards = zip(*res)
        obs = np.sum(np.array(states),axis=0)
        reward = np.sum(np.array(rewards)) + D[0]*(NATIONS+1-np.sum(action))

        obs, reward, self.time == DAYS-1, {}

    def render(self, mode='human', close='False'):
        pass
