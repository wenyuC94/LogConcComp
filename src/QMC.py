import numpy as np
import numpy.linalg as la
from numbers import Number
from scipy import stats


def sequ(n,d,lb=0,ub=1,method=None,randomized=True, random_state = 1, rng= None,return_likelihood = False):
    if rng is None:
        rng = np.random.RandomState(seed=random_state)
    if method is None:
        seq = np.zeros((n,d))
    elif method == "uniform":
        n0 = int(np.power(n,1/d))
        assert n == n0 **d, 'this method does not apply to n = %d'%n
        tmp_list = np.linspace(0,1,n0+1)
        tmp_list = (tmp_list[:-1]+tmp_list[1:])/2
        dim_grid_list = [list(tmp_list) for i in range(d)]
        seq = np.stack(np.meshgrid(*dim_grid_list)).T.reshape(n,d)
    elif method == "random":
        seq = rng.rand(n,d)
    
    if randomized and method!="random":
        seq += rng.rand(n,d)
        seq = np.mod(seq,1.0)
    seq = lb+seq*(ub-lb)
    if not return_likelihood:
        return seq
    else:
        if isinstance(ub-lb,Number):
            return seq, np.ones(n)/np.power(ub-lb,d)
        else:
            return seq, np.ones(n)/np.prod(ub-lb)
        
def generate_multivariate_normal(Z,mu,Sigma):
    C = la.cholesky(Sigma)
    X = Z@C.T + mu
    likelihood = np.exp(-np.sum(Z*Z,axis=1)/2)/np.power(2*np.pi,len(mu)/2)/np.sqrt(la.det(Sigma))
    return X,likelihood

def seqn(n,d,mu=0,Sigma=1,method=None,randomized=True, random_state = 1,rng=None,return_likelihood = False):
    if method != "random":
        seq = sequ(n,d,0,1,method,randomized,random_state,rng,False)
        seq = stats.norm.ppf(seq)
    else:
        if rng is None:
            rng = np.random.RandomState(seed=random_state)
        seq = rng.randn(n,d)
    
    if isinstance(mu,Number):
        mu = mu*np.ones(d)
    if isinstance(Sigma,Number):
        Sigma = np.diag(np.ones(d)*Sigma)
    
    if return_likelihood:
        return generate_multivariate_normal(seq, mu,Sigma)
    else:
        return generate_multivariate_normal(seq, mu,Sigma)[0]
    
class SeqGenerator:
    def __init__(self, method = None, randomized=True, seed = 1, rng = None):
        self.method = method
        self.randomized = randomized
        self.seed = seed
        if rng is None:
            self.rng = np.random.RandomState(seed=random_state)
        else:
            self.rng = rng
        self.cur_state = seed
    def rand(self,n,d, lb=0,ub=1,return_likelihood = False):
        res = sequ(n,d,lb=lb,ub=ub,method=self.method,randomized=self.randomized, random_state = self.cur_state, rng= self.rng,return_likelihood = return_likelihood)
        self.cur_state += n
        return res
    def randn(self,n,d,mu=0,Sigma=1,return_likelihood = False):
        res = seqn(n,d,mu=mu,Sigma=Sigma,method=self.method,randomized=self.randomized, random_state = self.cur_state,rng=self.rng,return_likelihood = return_likelihood)
        self.cur_state += n
        return res
        