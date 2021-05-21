import numpy as np
from numpy import linalg as la
from scipy.spatial import ConvexHull
import scipy.stats as st
from gurobipy import *

import copy
import time
from functools import reduce
from numbers import Number

from multiprocessing import Pool as pyPool
import multiprocessing
import io
import contextlib

import numba as nb
import warnings
warnings.filterwarnings('ignore')

import QMC
from utils import *

import json

from tqdm.notebook import tnrange
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt

from gurobipy import *


def randb(n,d,rng=None,random_state=None):
    rng = rng if rng is not None else np.random.RandomState(random_state)
    Z = rng.randn(n,d)
    Z/=np.apply_along_axis(la.norm,1,Z)[:,None]
    Z*= rng.rand(n,1)**(1/d)
    return Z


@nb.njit(cache=True)
def ab_to_phi(X,a,b):
    tmp = X@a.T+b
    return np_apply_along_axis(np.max, 1, tmp)


def convex_phi_from_eqns(equations, X):
    a = -equations[:,:-2]/equations[:,-2].reshape(-1,1)
    b = -equations[:,-1]/equations[:,-2]
    return ab_to_phi(X,a,b)


def ab_to_nu(X,a,b):
    tmp = X@a.T+b
    max_idx = tmp.argmax(axis=1)
    return -np.hstack([a[max_idx],b[max_idx].reshape(-1,1)])

@nb.njit(parallel=True,cache=True)
def feas_from_cvh(equations, X,threshold=1e10):
    N,d = X.shape
    nsimplex = equations.shape[0]
    _X = np.hstack((X,np.ones((N,1))))
    feas_list = np.full(N,False)
    if nsimplex*N*d <= threshold:
        feas_list = np_apply_along_axis(np.max,0,equations@_X.T) <= 0
        return list(feas_list)
    else:
        B = min(math.ceil(nsimplex*N*d/threshold),N)
        split_indices = split(N,B)
        start, end = 0, 0
        for ell in range(B):
            start = end
            end = split_indices[ell]
            x = _X[start:end]
            feas_list[start:end] = np_apply_along_axis(np.max,0,equations@x.T) <= 0
        return list(feas_list)
    
def func_piecewise_max_subg(f,g,X,w,a,b,thres = 1e10, whole = False):
    n,d = X.shape
    K,_ = a.shape
    if whole or n*K*(d+1) <= thres:
        tmp = X@a.T+b
        values = tmp.max(axis = 1)
        max_idx = tmp.argmax(axis = 1)
        subg_a = np.zeros_like(a)
        subg_b = np.zeros_like(b)
        np.add.at(subg_a, max_idx,X*w.reshape(-1,1)*g(values).reshape(-1,1))
        np.add.at(subg_b, max_idx,w*g(values))
        values = f(values)
        return values, subg_a, subg_b
    else:
        B = min(math.ceil(n*K*(d+1)/thres),n)
        values = np.zeros(n)
        subg_a = np.zeros_like(a)
        subg_b = np.zeros_like(b)
        split_indices = split(n,B)
        start, end = 0, 0
        for ell in range(B):
            start = end
            end = split_indices[ell]
            values[start:end], subg_a_tmp, subg_b_tmp = func_piecewise_max_subg_weights(f,g,X[start:end], w[start:end], a,b,thres=thres,whole=True)
            subg_a += subg_a_tmp
            subg_b += subg_b_tmp
        return values, subg_a, subg_b

identity = nb.njit(lambda x: x)
grad_identity = nb.njit(lambda x: np.ones_like(x))

expneg = nb.njit(lambda x: np.exp(-x))
grad_expneg = nb.njit(lambda x: -np.exp(-x))

    
@nb.njit(cache=True)
def calcJd(phi_d,sort=True,eps= 1e-3,factorial = np.array([1,1,2,6,24,120,720,5040,40320,362880,3628800,39916800,479001600])):
    if len(phi_d) ==1:
        return np.exp(phi_d[0])
    if not sort:
        phi_d = np.sort(phi_d)
    d = len(phi_d) - 1
    if phi_d[-1]-phi_d[0] < eps:
        phi_bar = np.mean(phi_d)
        phi_demean = phi_d - phi_bar
        sumPow2 = np.power(phi_demean,2).sum()
        sumPow3 = np.power(phi_demean,3).sum()
        return np.exp(phi_bar)*(1/factorial[d]+sumPow2/(2*factorial[d+2])+sumPow3/(3*factorial[d+3]))
    else:
        return (calcJd(phi_d[1:])-calcJd(phi_d[:-1]))/(phi_d[-1]-phi_d[0])


def calcExactIntegral(X, phi, grad = False,hypers = False,njobs = 0,verbose=0):
    if type(phi) == list:
        phi = np.array(phi)
    n = X.shape[0]
    Xphi = np.concatenate([X,phi.reshape(-1,1)],axis = 1)
    if la.norm(phi-phi[0]) <= 1e-8:
        cvh = ConvexHull(X)
        Delta = cvh.volume
        if grad:
            return Delta*np.exp(-phi[0]), np.ones(n)*Delta/n
        else:
            return Delta*np.exp(-phi[0])
        if hypers:
            return Delta*np.exp(-phi[0]), len(np.unique(cvh.simplices,axis=0))
    try:
        hull = ConvexHull(Xphi)
    except:
        hull = ConvexHull(Xphi, qhull_options='QJ')
    Xphi[:,-1] = 1
    active_simplices = hull.simplices[hull.equations[:,-2] <0]
    if verbose:
        print(active_simplices.shape)
    if njobs > 0:
        with Pool(njobs) as p:
            detlist = np.array(list(p.map(lambda simplex: (np.abs(la.det(Xphi[simplex]))),active_simplices)))
            Jdlist = np.array(list(p.map(lambda simplex:calcJd(-phi[simplex],sort=False), active_simplices)))
    else:
        detlist = np.array(list(map(lambda simplex: (np.abs(la.det(Xphi[simplex]))),active_simplices)))
        Jdlist = np.array(list(map(lambda simplex:calcJd(-phi[simplex],sort=False), active_simplices)))
    assert grad + hypers <= 1
    if grad:
        subgrad = np.ones(n)/n
        for i,simplex in enumerate(active_simplices):
            tmp = -phi[simplex]
            deti = detlist[i]
            if njobs > 0:
                Jdylist = np.array(list(p.map(lambda idx: calcJd(np.append(-tmp,-phi[idx]),sort=False), simplex)))
            else:
                Jdylist = np.array(list(map(lambda idx: calcJd(np.append(-tmp,-phi[idx]),sort=False), simplex)))
            subgrad[simplex] -= deti * Jdylist
        return (detlist*Jdlist).sum(), subgrad
    else:
        return (detlist*Jdlist).sum()
    if hypers:
        return (detlist*Jdlist).sum(), len(np.unique(active_simplices,axis=0 ))


class Iterate:
    def __init__(self,a = None, b = None, phi = None):
        assert (a is not None and b is not None) or (phi is not None)
        self.a = copy.deepcopy(a)
        self.b = copy.deepcopy(b)
        self.phi = copy.deepcopy(phi)
    def calcExactIntegral(self,sample):
        if self.phi is None:
            self.phi = _ab_to_phi(sample.X,self.a,self.b)
        self.exact_integral = sample.calcExactIntegral(self.phi,njobs = 24)
        return self.exact_integral
    

class Sample:
    N0dict = {2:100,3:50,4:24,5:12,6:8}
    N0dict_sparse = {2:50,3:24,4:12,5:8,6:6}
    def __init__(self, X=None, n=100,d=2,dist= "normal",random_state = 42,**kwargs):
        self.random = np.random.RandomState(random_state)
        if X is not None:
            self.X = X
            self.n, self.d = X.shape
        else:
            self.X, self.true_phi = self.generate_X(n,d,dist,**kwargs)
            self.n = n
            self.d = d
        self.dist = dist
        self.sample_str= '%d_%d_%s_seed%d'%(self.n,self.d,self.dist,random_state)
        
    
    def generate_X(self,n,d,dist= "normal",**kwargs):
        if dist == "normal":
            X = self.random.randn(n,d)
            true_phi = -np.sum(st.norm.logpdf(X),axis=1)
        elif dist == "uniform":
            X = self.random.rand(n,d)
            true_phi = np.zeros(n)
        elif dist == "laplace":
            loc = kwargs.get('loc', 0.0)
            scale = kwargs.get('scale',1.0)
            X = self.random.laplace(loc,scale,size=(n,d))
            true_phi = -np.sum(st.laplace.logpdf(X,loc=loc,scale=scale),axis=1)
        elif dist == "beta":
            a = kwargs.get('a',2.)
            b = kwargs.get('b',2.)
            X = self.random.beta(a,b,size=(n,d))
            true_phi = -np.sum(st.beta.logpdf(X,a,b),axis=1)
        elif dist == "dirichlet":
            alpha = kwargs.get('alpha',np.ones(d+1)*2)
            X = self.random.dirichlet(alpha, n)
            true_phi = -np.array([st.dirichlet.logpdf(x,alpha) for x in X])
            X = X[:,:-1]
        return X,true_phi
            
    
    def generate_feas_grid_mat(self, method="uniform", N = None, N0 = None, threshold=1e10,randomized=False,m=1,first=0,rng=None):
        rng = rng if rng is not None else self.random
        self.qmc_seq = QMC.SeqGenerator(method=method,randomized=randomized, seed=1,rng= rng)
        self.grid_method = method
        if not hasattr(self,"cvh"):
            self.cvh =  ConvexHull(self.X)
        if not hasattr(self,"min_d"):
            self.min_d = self.X.min(axis = 0)
            self.max_d = self.X.max(axis = 0)
        if not hasattr(self, "phat"):
            self.phat = self.cvh.volume/np.product(self.max_d-self.min_d)
        
        assert (N is not None or N0 is not None)
        N_total = 0
        if method =="uniform" or N is None:
            if N0 is not None:
                self.N0= N0
            else:
                self.N0 = self.N0dict.get(self.d, 6)
            N_total = self.N0**self.d
        else:
            N_total = int(N/self.phat*1.1)
            self.N = N
        grid_mat = self.qmc_seq.rand(N_total, self.d,lb = self.min_d, ub=self.max_d)
        feas_list = feas_from_cvh(self.cvh.equations,grid_mat,threshold)
        assert(len(feas_list) == N_total)
        if N is None:
            self.N = sum(feas_list)
        else:
            self.N = min(N,sum(feas_list))
        self.grid_mat = grid_mat[np.array(feas_list)][:self.N]
        
        self.Delta = self.cvh.volume
        if method == "uniform" and m != 1:
            self.grid_mat = self.grid_mat[first::m,:]
            self.N= self.grid_mat.shape[0]

    
    def generate_new_random_feas_grid_mat(self, N1 = None, threshold=1e10,rng=None):
        if not hasattr(self,"grid_method") or self.grid_method != "random":
            self.grid_method = "random"
        if not hasattr(self,"cvh"):
            self.cvh =  ConvexHull(self.X)
            self.Delta = self.cvh.volume
        if not hasattr(self,"min_d"):
            self.min_d = self.X.min(axis = 0)
            self.max_d = self.X.max(axis = 0)
        if not hasattr(self, "phat"):
            self.phat = self.cvh.volume/np.product(self.max_d-self.min_d)
        if rng is None:
            rng = self.random
        N_total = int(N1/self.phat*1.1)
        grid_mat = self.min_d+rng.rand(N_total,self.d)*(self.max_d-self.min_d)
        feas_list = feas_from_cvh(self.cvh.equations,grid_mat,threshold)
        assert(len(feas_list) == N_total)
        N1 = min(N1,sum(feas_list))
        return grid_mat[np.array(feas_list)][:N1]
        

    
    def generate_X1(self):
        self.X1 = np.hstack((self.X,np.ones((self.n,1))))
    
    def calcExactIntegral(self,phi,grad = False, njobs = 0,verbose = 0):
        return calcExactIntegral(self.X,phi,grad,njobs,verbose)

def solve_LPs_gurobi(phi,X1,X_list):
    N1, d = X_list.shape
    n = X1.shape[0]
    x = np.zeros(d+1)
    x[-1] = 1
    model = Model()
    model.Params.OutputFlag=0
    model.Params.presolve = 1
    alpha = model.addVars(n,lb=0,name="alpha")
    conv_comb = model.addConstrs(( quicksum(alpha[i]*X1[i,k] for i in range(n)) == x[k] for k in range(d+1)))
    model.setObjective(quicksum(phi[i]*alpha[i] for i in range(n)),GRB.MINIMIZE)
    model.update()
    integral = 0
    grad_int = np.zeros(n)
    for ell in range(N1):
        x = X_list[ell,:]
        for k in range(d):
            conv_comb[k].rhs = x[k]
        model.optimize()
        integral += np.exp(-model.ObjVal)
        grad_int -= np.exp(-model.ObjVal)*np.array(model.X)

    return integral, grad_int

def solve_QPs_gurobi(phi,X1,X_list,u):
    N1, d = X_list.shape
    n = X1.shape[0]
    x = np.zeros(d+1)
    x[-1] = 1
    model = Model()
    model.Params.OutputFlag=0
    model.Params.BarHomogeneous = 1
    model.Params.presolve = 1
    alpha = model.addVars(n,lb=0,name="alpha")
    conv_comb = model.addConstrs(( quicksum(alpha[i]*X1[i,k] for i in range(n)) == x[k] for k in range(d+1)))
    model.setObjective(quicksum(phi[i]*alpha[i]+u/2*(alpha[i]-1/n)*(alpha[i]-1/n) for i in range(n)),GRB.MINIMIZE)
    model.update()
    integral = 0
    grad_int = np.zeros(n)
    for ell in range(N1):
        x = X_list[ell,:]
        for k in range(d):
            conv_comb[k].rhs = x[k]
        model.optimize()
        integral += np.exp(-model.ObjVal)
        grad_int -= np.exp(-model.ObjVal)*np.array(model.X)

    return integral, grad_int

def get_all_from_phi_LP(iterate,X1,Delta,w,grid_mat,normalize=True,njobs=24):
    N = grid_mat.shape[0]
    phi = iterate.phi
    if njobs == 1:
        res = solve_LPs_gurobi(phi,X1,grid_mat)
        integral = res[0]
        grad_int = res[1]
    else:
        with pyPool(njobs) as pool:
            res = pool.starmap(solve_LPs_gurobi, zip([phi]*njobs,[X1]*njobs, np.array_split(grid_mat,njobs)))
        integral = np.sum([x[0] for x in res])
        grad_int = np.sum(np.array([x[1] for x in res]),axis=0)        
    integral *= (Delta/N)
    grad_int *= (Delta/N)
    
    iterate.phi += (np.log(integral) if normalize else 0)
    iterate.integral = 1
    iterate.grad_phi = grad_int / (integral if normalize else 1) + w
    iterate.obj = phi@w + iterate.integral
    
    return iterate

def get_all_from_phi_QP(iterate,X1,Delta,w,grid_mat,u,normalize=True,njobs=24):
    N = grid_mat.shape[0]
    phi = iterate.phi
    if njobs == 1:
        res = solve_QPs_gurobi(phi,X1,grid_mat,u)
        integral = res[0]
        grad_int = res[1]
    else:
        with pyPool(njobs) as pool:
            res = pool.starmap(solve_QPs_gurobi, zip([phi]*njobs,[X1]*njobs, np.array_split(grid_mat,njobs),[u]*njobs))
        integral = np.sum([x[0] for x in res])
        grad_int = np.sum(np.array([x[1] for x in res]),axis=0)        
    integral *= (Delta/N)
    grad_int *= (Delta/N)
    
    iterate.phi += (np.log(integral) if normalize else 0)
    iterate.integral = 1 if normalize else integral
    iterate.grad_phi = grad_int / (integral if normalize else 1) + w
    iterate.obj = phi@w + iterate.integral
    
    return iterate

def get_all_from_ab(iterate,X,grid_mat,Delta,w,normalize=True,thres=1e10):
    n,d = X.shape
    N,d = grid_mat.shape
    a = iterate.a
    b = iterate.b
    phi, grad_sum_a, grad_sum_b = func_piecewise_max_subg(identity,grad_identity,X,w,a,b)
    likelihood, grad_int_a, grad_int_b = func_piecewise_max_subg(expneg,grad_expneg,grid_mat,np.ones(N),a,b)
    integral = np.sum(likelihood) * Delta / N
    grad_int_a *= (Delta/N)
    grad_int_b *= (Delta/N)
    
    iterate.b += (np.log(integral) if normalize else 0)
    iterate.phi = phi + (np.log(integral) if normalize else 0)
    iterate.integral = 1 if normalize else integral
    iterate.obj = iterate.phi@w + iterate.integral
    iterate.grad_a = grad_sum_a + grad_int_a / (integral if normalize else 1)
    iterate.grad_b = grad_sum_b + grad_int_b / (integral if normalize else 1)
    
    return iterate


class NCLCD:
    def __init__(self, sample:Sample, w = None, K = None, K0 = 10, maxIters = 100,
                 initStepSize = 1, stepSizeMode = "sqrt-decay", 
                 tol = 1e-5, normalize = True, random_state = 42, plot=True):
        self.sample = sample
        self.w = w if w is not None else np.ones(self.sample.n)/self.sample.n
        assert K is not None or K0 is not None
        self.K = K if K is not None else sample.d * K0
        self.iterates = []
        self.maxIters = maxIters
        self.initStepSize = initStepSize
        self.stepSizeMode = "sqrt-decay" 
        self.normalize = normalize
        self.tol = tol
        self.random = np.random.RandomState(random_state)
        self.plot = plot
    
    
    
    @nb.jit(nopython=False)
    def subgradient(self, a = None, b = None):
        self.algo_name = "nonconvex_subgradient"
        if a is None:
            a = self.random.randn(self.K, self.sample.d)
            b = self.random.randn(self.K)
        
        self.algo_str = "nonconvex_subgradient_%d_%d_%.0e_%d_uniform"%(self.K, self.maxIters,self.tol, self.sample.N0)
        start_t = time.time()
        print("algorithm started...")
        last = Iterate(a=a,b=b)
        get_all_from_ab(last,self.sample.X,self.sample.grid_mat,self.sample.Delta,self.w,self.normalize)
        last.time = time.time()-start_t
        min_obj = last.obj
        arg_min = 0
        increase_cnt = 0
        self.iterates.append(last)
        for t in tnrange(1,self.maxIters+1):
            a = last.a - self.initStepSize * last.grad_a / la.norm(last.grad_a) / np.sqrt(t)
            b = last.b - self.initStepSize * last.grad_b / la.norm(last.grad_b) / np.sqrt(t)
            cur = Iterate(a=a,b=b)
            get_all_from_ab(cur,self.sample.X,self.sample.grid_mat,self.sample.Delta,self.w,self.normalize)
            cur.time = time.time()-start_t
            self.iterates.append(cur)
            if np.abs(min_obj - cur.obj) <= self.tol:
                print(min_obj,cur.obj)
                break
            if cur.obj > min_obj+self.tol:
                increase_cnt += 1
                if increase_cnt == 20:
                    break
            else:
                increase_cnt = 0
            if cur.obj < min_obj:
                arg_min = t
                min_obj = cur.obj
            last = cur
        self.disc_arg_min = arg_min
        self.min_disc_obj = min_obj
        self.runtime = time.time() - start_t
        self.final_phi = self.iterates[self.disc_arg_min].phi
        self.final_time = self.iterates[self.disc_arg_min].time
        print("algorithm finished!\n")
        print("running time: ", "\033[1m",time_to_string(self.runtime),"\033[0;0m","\n" )
        
        self.disc_obj_times = [iterate.time for iterate in self.iterates]
        self.disc_objs = [iterate.obj for iterate in self.iterates]
        
        print("min discretized obj achieved: ", "\033[1m",self.min_disc_obj,"\033[0;0m")
        
        if self.plot:
            plt.figure(figsize=(16,9))
            plt.scatter(self.iterates[self.disc_arg_min].time, self.min_disc_obj, c="red",zorder = 1)
            plt.plot(self.disc_obj_times,self.disc_objs,color="orange",zorder=0)
            plt.xlabel("time")
            plt.ylabel("obj")
            plt.show()
        
        
    def calcExactFinalObjective(self,phi = None,verbose=0):
        if phi is None:
            integral = self.sample.calcExactIntegral(self.final_phi,verbose=verbose)
            return self.final_phi@self.w + np.log(integral) + 1 
        else:
            integral = self.sample.calcExactIntegral(phi,verbose=verbose)
            return phi@self.w + np.log(integral) + 1 
    
    def to_dict(self,contents = "all"):
        d = dict()
        d2 = dict()
        if contents in ("all","info"):
            for key in self.__dict__:
                if key not in ['sample','iterates','random','solver_class','solver','ws_duals','solver_method','lbmodel','phi','v','rad','rad_constr','ub_constrs','lb_constrs','int_pos','status','stepModel']:
                    if key not in ['final_phi','w']:
                        d[key] = getattr(self,key)
                    else:
                        d[key] = getattr(self,key).tolist()
            d['n'] = self.sample.n
            d['d'] = self.sample.d
        if contents in ("all","hist"):
            tmp_iterate = self.iterates[self.disc_arg_min]
            d2['phi_hist'] = [it.phi.tolist() for it in self.iterates]
            d2['a_hist'] = [it.a.tolist() for it in self.iterates]
            d2['b_hist'] = [it.b.tolist() for it in self.iterates]
            d2['grad_a_hist'] = [it.grad_a.tolist() for it in self.iterates]
            d2['grad_b_hist'] = [it.grad_b.tolist() for it in self.iterates]
        
        if contents =="info":
            return d
        elif contents =="hist":
            return d2
        elif contents=="all":
            d.update(d2)
            return d
            
    def to_json(self,path):
        d = dict()
        d2 = dict()
        for key in self.__dict__:
            if key not in ['sample','iterates','random','solver_class','solver','ws_duals','solver_method','lbmodel','phi','v','rad','rad_constr','ub_constrs','lb_constrs','int_pos','status','stepModel']:
                if key not in ['final_phi','w']:
                    d[key] = getattr(self,key)
                else:
                    d[key] = getattr(self,key).tolist()
                
        d['n'] = self.sample.n
        d['d'] = self.sample.d
        tmp_iterate = self.iterates[self.disc_arg_min]
        d2['phi_hist'] = [it.phi.tolist() for it in self.iterates]
        d2['a_hist'] = [it.a.tolist() for it in self.iterates]
        d2['b_hist'] = [it.b.tolist() for it in self.iterates]
        d2['grad_a_hist'] = [it.grad_a.tolist() for it in self.iterates]
        d2['grad_b_hist'] = [it.grad_b.tolist() for it in self.iterates]
        with open(path+"info/%s_%s_info.json"%(self.sample.sample_str,self.algo_str),'w') as f:
            json.dump(d,f)
        with open(path+"hist/%s_%s_hist.json"%(self.sample.sample_str,self.algo_str),'w') as f:
            json.dump(d2,f)
        with open(path+"soln/%s_%s_soln.npy"%(self.sample.sample_str,self.algo_str),'wb') as f:
            np.save(f, self.final_phi)
            

class ConvexLCD:
    def __init__(self, sample:Sample, w = None, maxIters = 128, maxtime = 14400, random_state = 42, njobs = -1, verbose = 0, normalize = False, evaluation=True, plot=True, **kwargs):
        self.sample = sample
        self.w = w if w is not None else np.ones(self.sample.n)/self.sample.n
        self.iterates = []
        self.maxIters = maxIters
        self.maxtime = maxtime
        self.random = np.random.RandomState(random_state)
        self.random_state = random_state
        self.sample.generate_X1()
        self.verbose=  verbose
        self.normalize = normalize
        self.evaluation = evaluation
        self.plot = plot
        if njobs == 0:
            self.parallel = False
        else:
            self.parallel = True
            self.njobs = njobs if njobs > 0 else min(multiprocessing.cpu_count(),24)
    
    def compute_phi_init(self, init_method, **init_kwargs):
        if init_method == "uniform":
            phi = np.log(self.sample.Delta)*np.ones(self.sample.n)
            self.init_suffix = "_uniform_init"
        elif init_method == "kde":
            phi = -np.log(st.gaussian_kde(self.sample.X.T).evaluate(self.sample.X.T))
            self.init_suffix = "_kde_init"
        elif init_method == "nonconvex":
            N0 = init_kwargs.get("N0", self.sample.N0dict.get(self.sample.d,6))
            K = init_kwargs.get("K", 100)
            tol = init_kwargs.get("tol", 1e-8)
            maxIters = init_kwargs.get("maxIters", 1000)
            self.sample.generate_feas_grid_mat(method ="uniform",N0=N0)
            nclcd = NCLCD(self.sample,K=K,maxIters = maxIters,tol=0.1**tol)
            nclcd.subgradient()
            phi = nclcd.iterates[nclcd.disc_arg_min].phi
            self.init_suffix = "_nc_init_%d_%d_%.0e_%d"%(K, maxIters, tol, self.sample.N0)
        elif init_method =="given":
            phi = init_kwargs.get("phi_start")
            self.init_suffix = init_kwargs.get("init_suffix")
#         self.phi_start = phi
        return phi
    
    def subgradient_approx(self,init_method=None,init_kwargs = dict(),initStepSize=5,stepSizeMode= "sqrt-decay-length",Nlist=None,N0list=None,thres_list= None,grid_method="uniform"):
        self.algo_name = "subgradient (approx)"
        
        if N0list is None:
            using_N = True
        else:
            Nlist = [int((N0list[i]**self.sample.d)*self.sample.phat) for i in range(len(N0list))]
            using_N = False
        if thres_list is None:
            maxStages = int(np.ceil(np.log2(self.maxIters)))
            thres_list = [2**i for i in range(maxStages-len(Nlist)+1,maxStages+1)]
            thres_list[-1] = min(thres_list[-1],self.maxIters)
        self.N0list = N0list
        self.Nlist = Nlist
        if grid_method == "uniform":
            self.algo_str = "subgradient_Riemann_%d_%s_%d_%d_%d"%(initStepSize,stepSizeMode, self.maxIters, len(N0list),max(N0list))
        elif not using_N:
            self.algo_str = "subgradient_rndapprox_%d_%s_%d_%d_%d"%(initStepSize,stepSizeMode, self.maxIters, len(N0list),max(N0list))
        else:
            self.algo_str = "subgradient_rndapprox_%d_%s_%d_%d_%d"%(initStepSize,stepSizeMode, self.maxIters, len(Nlist),max(Nlist))
        self.algo_str += '_Copy%d'%self.random_state
        phi = self.compute_phi_init(init_method,**init_kwargs)
        self.algo_str += self.init_suffix

        self.thres_list = thres_list
        self.initStepSize = initStepSize
        self.stepSizeMode = stepSizeMode
        totalStages = len(Nlist)
        cur_stage = 0
        cur_N = Nlist[cur_stage]
        cur_thres = thres_list[cur_stage]
        if using_N:
            self.sample.generate_feas_grid_mat(method=grid_method, N = Nlist[cur_stage],rng=self.random)
        else:
            self.sample.generate_feas_grid_mat(method=grid_method,N0=N0list[cur_stage],rng=self.random)
            cur_N = self.sample.N
            Nlist[cur_stage] = cur_N
        print("algorithm started...")
        start_t = time.time()
        for t in tnrange(self.maxIters+1):
            if t > cur_thres:
                cur_stage += 1
                cur_N = Nlist[cur_stage]
                cur_thres = thres_list[cur_stage]
                print("Iteration",t,":")
                if using_N:
                    print("N changed to %d"%cur_N)
                    self.sample.generate_feas_grid_mat(method=grid_method, N = Nlist[cur_stage],rng=self.random)
                else:
                    print("N0 changed to %d"%N0list[cur_stage])
                    self.sample.generate_feas_grid_mat(method=grid_method,N0=N0list[cur_stage],rng=self.random)
                    cur_N = self.sample.N
                    Nlist[cur_stage] = cur_N
                    print("N changed to %d"%cur_N)
            
            cur = Iterate(phi = phi)
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                get_all_from_phi_LP(cur, self.sample.X1, self.sample.Delta, self.w, self.sample.grid_mat, normalize = self.normalize, njobs= self.njobs )
            cur.time = time.time()-start_t
            if self.stepSizeMode == "sqrt-decay-length":
                phi = cur.phi - self.initStepSize*cur.grad_phi / la.norm(cur.grad_phi) / np.sqrt(t+1)
            elif self.stepSizeMode == "sqrt-decay-size":
                phi = cur.phi - self.initStepSize*cur.grad_phi / np.sqrt(t+1)
            elif self.stepSizeMode == "constant-length":
                phi = cur.phi - self.initStepSize*cur.grad_phi / la.norm(cur.grad_phi)
            elif self.stepSizeMode == "constant-size":
                phi = cur.phi - self.initStepSize*cur.grad_phi
            self.iterates.append(cur)
            if cur.time > self.maxtime:
                print("algorithm terminated due to time limit\n")
                break
            
        self.Nlist = Nlist
        self.runtime = time.time() - start_t
        print("algorithm finished!\n")
        print("running time: ", "\033[1m",time_to_string(self.runtime),"\033[0;0m","\n" )
        
        self.disc_obj_times = [iterate.time for iterate in self.iterates]
        self.disc_objs = [iterate.obj for iterate in self.iterates]
        self.disc_normalized_objs = [np.mean(iterate.phi)+np.log(iterate.integral)+1 for iterate in self.iterates]
        if self.evaluation:
            print("function evaluation started...")
            self.real_objs = []
            self.real_integrals = []
            self.obj_times = []
            self.arg_min = 0
            self.eval_iters=1
            self.min_obj = float('inf')
            for i in tnrange(len(self.iterates)):
                if i % self.eval_iters == 0:
                    phi = self.iterates[i].phi
                    obj, integral = self.calcExactFinalObjective(phi,verbose=0)
                    self.real_objs.append(obj)
                    self.real_integrals.append(integral)
                    self.obj_times.append(self.iterates[i].time)
                    if obj < self.min_obj:
                        self.min_obj = obj
                        self.arg_min = i
            print("function evaluation finished!\n\n")


            self.final_phi = self.iterates[self.arg_min].phi
            self.final_time = self.iterates[self.arg_min].time
            print("min obj achieved: ", "\033[1m",self.min_obj,"\033[0;0m")
            
            if self.plot:
                plt.figure(figsize=(16,9))
                plt.scatter(self.final_time, self.min_obj, c="red",zorder = 1)
                plt.plot(self.obj_times,self.real_objs,color ="orange",zorder=0)
                plt.xlabel("time")
                plt.ylabel("obj")
                plt.show()

    def subgradient_stoch(self,init_method=None,init_kwargs = dict(),initStepSize=5,stepSizeMode= "sqrt-decay-length",Nlist= [5000,10000,20000,40000,80000],thres_list= None):
        self.algo_name = "subgradient (stochastic)"
        self.algo_str = "subgradient_stochastic_%d_%s_%d_%d_%d"%(initStepSize,stepSizeMode, self.maxIters, len(Nlist),max(Nlist))
        self.algo_str += '_Copy%d'%self.random_state
        self.Nlist = Nlist
        phi = self.compute_phi_init(init_method,**init_kwargs)
        self.algo_str += self.init_suffix
        
        if thres_list is None:
            maxStages = int(np.ceil(np.log2(self.maxIters)))
            thres_list = [2**i for i in range(maxStages-len(Nlist)+1,maxStages+1)]
            thres_list[-1] = min(thres_list[-1],self.maxIters)

        self.thres_list = thres_list
        self.initStepSize = initStepSize
        self.stepSizeMode = stepSizeMode
        totalStages = len(Nlist)
        cur_stage = 0
        cur_N = Nlist[cur_stage]
        cur_thres = thres_list[cur_stage]
        print("algorithm started...")
        start_t = time.time()
        for t in tnrange(self.maxIters+1):
            if t > cur_thres:
                cur_stage += 1
                cur_N = Nlist[cur_stage]
                cur_thres = thres_list[cur_stage]
                print("Iteration",t,":")
                print("N changed to %d"%cur_N)
            grid_mat = self.sample.generate_new_random_feas_grid_mat(cur_N, rng = self.random)
            
            cur = Iterate(phi = phi)
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                get_all_from_phi_LP(cur, self.sample.X1, self.sample.Delta, self.w, grid_mat, normalize = self.normalize, njobs= self.njobs )
            cur.time = time.time()-start_t
            if self.stepSizeMode == "sqrt-decay-length":
                phi = cur.phi - self.initStepSize*cur.grad_phi / la.norm(cur.grad_phi) / np.sqrt(t+1)
            elif self.stepSizeMode == "sqrt-decay-size":
                phi = cur.phi - self.initStepSize*cur.grad_phi / np.sqrt(t+1)
            elif self.stepSizeMode == "constant-length":
                phi = cur.phi - self.initStepSize*cur.grad_phi / la.norm(cur.grad_phi)
            elif self.stepSizeMode == "constant-size":
                phi = cur.phi - self.initStepSize*cur.grad_phi
            self.iterates.append(cur)
            if cur.time > self.maxtime:
                print("algorithm terminated due to time limit\n")
                break
        self.Nlist = Nlist
        self.runtime = time.time() - start_t
        print("algorithm finished!\n")
        print("running time: ", "\033[1m",time_to_string(self.runtime),"\033[0;0m","\n" )
        
        self.disc_obj_times = [iterate.time for iterate in self.iterates]
        self.disc_objs = [iterate.obj for iterate in self.iterates]
        self.disc_normalized_objs = [np.mean(iterate.phi)+np.log(iterate.integral)+1 for iterate in self.iterates]
        if self.evaluation:
            print("function evaluation started...")
            self.real_objs = []
            self.real_integrals = []
            self.obj_times = []
            self.arg_min = 0
            self.eval_iters=1
            self.min_obj = float('inf')
            for i in tnrange(len(self.iterates)):
                if i % self.eval_iters == 0:
                    phi = self.iterates[i].phi
                    obj, integral = self.calcExactFinalObjective(phi,verbose=0)
                    self.real_objs.append(obj)
                    self.real_integrals.append(integral)
                    self.obj_times.append(self.iterates[i].time)
                    if obj < self.min_obj:
                        self.min_obj = obj
                        self.arg_min = i
            print("function evaluation finished!\n\n")


            self.final_phi = self.iterates[self.arg_min].phi
            self.final_time = self.iterates[self.arg_min].time

            print("min obj achieved: ", "\033[1m",self.min_obj,"\033[0;0m")
            if self.plot:
                plt.figure(figsize=(16,9))
                plt.scatter(self.final_time, self.min_obj, c="red",zorder = 1)
                plt.plot(self.obj_times,self.real_objs,color ="orange",zorder=0)
                plt.xlabel("time")
                plt.ylabel("obj")
                plt.show()

    
    def randomized_smoothing_stoch(self,init_method=None,init_kwargs = dict(),D=2,sigma=1,beta=0.25,Nlist= [5000,10000,20000,40000,80000],thres_list= None,eta_mode = "constant"):
        self.algo_name = "randomized_smoothing (stochastic)"
        self.algo_str = "randomized_smoothing_stochastic_%d_%.0e_%.2f_%d_%d_%d"%(D,sigma,beta, self.maxIters, len(Nlist),max(Nlist))
        self.algo_str += '_Copy%d'%self.random_state
        phi = self.compute_phi_init(init_method,**init_kwargs)
        self.algo_str += self.init_suffix
        self.beta = beta
        u = D*(self.sample.n**(beta))/2
        if thres_list is None:
            maxStages = int(np.ceil(np.log2(self.maxIters)))
            thres_list = [2**i for i in range(maxStages-len(Nlist)+1,maxStages+1)]
            thres_list[-1] = min(thres_list[-1],self.maxIters)
        self.thres_list = thres_list
        self.Nlist = Nlist
        stage_iters = np.r_[thres_list[0], np.diff(thres_list)]
        M = np.sqrt(np.sum([stage_iters[i]/Nlist[i] for i in range(len(Nlist))]))
        eta = sigma*M/D
        self.u = u
        self.D = D
        self.M= M
        self.eta= eta
        self.sigma = sigma
        self.eta_mode = eta_mode
        if self.eta_mode == "duchi":
            self.algo_str += "_duchi"
        totalStages = len(Nlist)
        cur_stage = 0
        cur_N = Nlist[cur_stage]
        cur_thres = thres_list[cur_stage]
        phi_x = phi
        phi_y = phi
        phi_z = phi
        s = np.zeros(self.sample.n)
        theta_old= 1
        theta = 1
        print("algorithm started...")
        start_t = time.time()
        for t in tnrange(self.maxIters+1):
            if t > cur_thres:
                cur_stage += 1
                cur_N = Nlist[cur_stage]
                cur_thres = thres_list[cur_stage]
                print("Iteration",t,":")
                print("N changed to %d"%cur_N)
            grid_mat = self.sample.generate_new_random_feas_grid_mat(cur_N, rng = self.random)
            ut = u*theta_old
            phi_y = (1-theta_old)*phi_x+theta_old*phi_z
            cur = Iterate(phi = phi_y+ut*randb(1,self.sample.n, rng = self.random).reshape(-1))
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                get_all_from_phi_LP(cur, self.sample.X1, self.sample.Delta, self.w, grid_mat, normalize = self.normalize, njobs= self.njobs )
            cur.time = time.time()-start_t
            s += cur.grad_phi/theta_old
            theta = 2/(1+np.sqrt(1+4/theta_old**2))
            cur.phi_x = phi_x
            cur.phi_y = phi_y
            cur.phi_z = phi_z
            if self.eta_mode == "duchi":
                eta = self.eta * np.sqrt(t+1)
            phi_z = phi - s*theta/(sigma*np.sqrt(self.sample.n)/u+eta)
            phi_x = (1-theta_old)*phi_x+theta_old*phi_z
            theta_old = theta
            self.iterates.append(cur)
            if cur.time > self.maxtime:
                print("algorithm terminated due to time limit\n")
                break
        self.runtime = time.time() - start_t
        print("algorithm finished!\n")
        print("running time: ", "\033[1m",time_to_string(self.runtime),"\033[0;0m","\n" )
        
        self.disc_obj_times = [iterate.time for iterate in self.iterates]
        self.disc_objs = [iterate.obj for iterate in self.iterates]
        self.disc_normalized_objs = [np.mean(iterate.phi)+np.log(iterate.integral)+1 for iterate in self.iterates]
        
        if self.evaluation:
            print("function evaluation started...")
            self.real_objs = []
            self.real_integrals = []
            self.obj_times = []
            self.arg_min = 0
            self.eval_iters=1
            self.min_obj = float('inf')
            for i in tnrange(len(self.iterates)):
                if i % self.eval_iters == 0:
                    phi = self.iterates[i].phi_x
                    obj, integral = self.calcExactFinalObjective(phi,verbose=0)
                    self.real_objs.append(obj)
                    self.real_integrals.append(integral)
                    self.obj_times.append(self.iterates[i].time)
                    if obj < self.min_obj:
                        self.min_obj = obj
                        self.arg_min = i
            print("function evaluation finished!\n\n")


            self.final_phi = self.iterates[self.arg_min].phi
            self.final_time = self.iterates[self.arg_min].time
            
            print("min obj achieved: ", "\033[1m",self.min_obj,"\033[0;0m")
            if self.plot:
                plt.figure(figsize=(16,9))
                plt.scatter(self.final_time, self.min_obj, c="red",zorder = 1)
                plt.plot(self.obj_times,self.real_objs,color ="orange",zorder=0)
                plt.xlabel("time")
                plt.ylabel("obj")
                plt.show()
    
    def randomized_smoothing_approx(self,init_method=None,init_kwargs = dict(),D=2,sigma=1,beta=0.25,Nlist=None,N0list=None,thres_list= None,grid_method="uniform",eta_mode = "constant"):
        self.algo_name = "randomized_smoothing (approx)"

        
        u = D*(self.sample.n**(beta))/2
        self.beta = beta
        if N0list is None:
            using_N = True
        else:
            Nlist = [int((N0list[i]**self.sample.d)*self.sample.phat) for i in range(len(N0list))]
            using_N = False
        if thres_list is None:
            maxStages = int(np.ceil(np.log2(self.maxIters)))
            thres_list = [2**i for i in range(maxStages-len(Nlist)+1,maxStages+1)]
            thres_list[-1] = min(thres_list[-1],self.maxIters)
        self.thres_list = thres_list
        self.N0list = N0list
        self.Nlist = Nlist
        if grid_method == "uniform":
            self.algo_str = "randomized_smoothing_Riemann_%d_%.0e_%.2f_%d_%d_%d"%(D,sigma,beta, self.maxIters, len(N0list),max(N0list))
        elif not using_N:
            self.algo_str = "randomized_smoothing_rndapprox_%d_%.0e_%.2f_%d_%d_%d"%(D,sigma,beta, self.maxIters, len(N0list),max(N0list))
        else:
            self.algo_str = "randomized_smoothing_rndapprox_%d_%.0e_%.2f_%d_%d_%d"%(D,sigma,beta, self.maxIters, len(Nlist),max(Nlist))
        self.algo_str += '_Copy%d'%self.random_state
        phi = self.compute_phi_init(init_method,**init_kwargs)
        self.algo_str += self.init_suffix
            
        stage_iters = np.r_[thres_list[0], np.diff(thres_list)]
        M = np.sqrt(np.sum([stage_iters[i]/Nlist[i] for i in range(len(Nlist))]))
        eta = sigma*M/D
        self.u = u
        self.D = D
        self.eta= eta
        self.M= M
        self.sigma = sigma
        self.eta_mode = eta_mode
        if self.eta_mode == "duchi":
            self.algo_str += "_duchi"
        totalStages = len(Nlist)
        cur_stage = 0
        cur_N = Nlist[cur_stage]
        cur_thres = thres_list[cur_stage]
        if using_N:
            self.sample.generate_feas_grid_mat(method=grid_method, N = Nlist[cur_stage],rng=self.random)
        else:
            self.sample.generate_feas_grid_mat(method=grid_method,N0=N0list[cur_stage],rng=self.random)
            cur_N = self.sample.N
            Nlist[cur_stage] = cur_N
        phi_x = phi
        phi_y = phi
        phi_z = phi
        s = np.zeros(self.sample.n)
        theta_old= 1
        theta = 1
        print("algorithm started...")
        start_t = time.time()
        for t in tnrange(self.maxIters+1):
            if t > cur_thres:
                cur_stage += 1
                cur_N = Nlist[cur_stage]
                cur_thres = thres_list[cur_stage]
                print("Iteration",t,":")
                if using_N:
                    print("N changed to %d"%cur_N)
                    self.sample.generate_feas_grid_mat(method=grid_method, N = Nlist[cur_stage],rng=self.random)
                else:
                    print("N0 changed to %d"%N0list[cur_stage])
                    self.sample.generate_feas_grid_mat(method=grid_method,N0=N0list[cur_stage],rng=self.random)
                    cur_N = self.sample.N
                    Nlist[cur_stage] = cur_N
                    print("N changed to %d"%cur_N)
            ut = u*theta_old
            phi_y = (1-theta_old)*phi_x+theta_old*phi_z
            cur = Iterate(phi = phi_y+ut*randb(1,self.sample.n,rng = self.random).reshape(-1))
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                get_all_from_phi_LP(cur, self.sample.X1, self.sample.Delta, self.w, self.sample.grid_mat, normalize = self.normalize, njobs= self.njobs )
            cur.time = time.time()-start_t
            s += cur.grad_phi/theta_old
            theta = 2/(1+np.sqrt(1+4/theta_old**2))
            cur.phi_x = phi_x
            cur.phi_y = phi_y
            cur.phi_z = phi_z
            if self.eta_mode == "duchi":
                eta = self.eta * np.sqrt(t+1)
            phi_z = phi - s*theta/(sigma*np.sqrt(self.sample.n)/u+eta)
            phi_x = (1-theta_old)*phi_x+theta_old*phi_z
            theta_old = theta
            self.iterates.append(cur)
            if cur.time > self.maxtime:
                print("algorithm terminated due to time limit\n")
                break
        self.Nlist = Nlist
        self.runtime = time.time() - start_t
        print("algorithm finished!\n")
        print("running time: ", "\033[1m",time_to_string(self.runtime),"\033[0;0m","\n" )
        
        self.disc_obj_times = [iterate.time for iterate in self.iterates]
        self.disc_objs = [iterate.obj for iterate in self.iterates]
        self.disc_normalized_objs = [np.mean(iterate.phi)+np.log(iterate.integral)+1 for iterate in self.iterates]
        
        if self.evaluation:
            print("function evaluation started...")
            self.real_objs = []
            self.real_integrals = []
            self.obj_times = []
            self.arg_min = 0
            self.eval_iters=1
            self.min_obj = float('inf')
            for i in tnrange(len(self.iterates)):
                if i % self.eval_iters == 0:
                    phi = self.iterates[i].phi_x
                    obj, integral = self.calcExactFinalObjective(phi,verbose=0)
                    self.real_objs.append(obj)
                    self.real_integrals.append(integral)
                    self.obj_times.append(self.iterates[i].time)
                    if obj < self.min_obj:
                        self.min_obj = obj
                        self.arg_min = i
            print("function evaluation finished!\n\n")


            self.final_phi = self.iterates[self.arg_min].phi
            self.final_time = self.iterates[self.arg_min].time

            print("min obj achieved: ", "\033[1m",self.min_obj,"\033[0;0m")
            if self.plot:
                plt.figure(figsize=(16,9))
                plt.scatter(self.final_time, self.min_obj, c="red",zorder = 1)
                plt.plot(self.obj_times,self.real_objs,color ="orange",zorder=0)
                plt.xlabel("time")
                plt.ylabel("obj")
                plt.show()
    

    def nesterov_smoothing_stoch(self,init_method=None,init_kwargs = dict(),D=2,sigma=1,C1=1,Nlist= [5000,10000,20000,40000,80000],thres_list= None,eta_mode="constant"):
        self.algo_name = "nesterov_smoothing (stochastic)"
        self.algo_str = "nesterov_smoothing_stochastic_%d_%.0e_%.2f_%d_%d_%d"%(D,sigma,C1, self.maxIters, len(Nlist),max(Nlist))
        self.algo_str += '_Copy%d'%self.random_state
        phi = self.compute_phi_init(init_method,**init_kwargs)
        self.algo_str += self.init_suffix
        self.C1 = C1
        u = D/2*C1
        if thres_list is None:
            maxStages = int(np.ceil(np.log2(self.maxIters)))
            thres_list = [2**i for i in range(maxStages-len(Nlist)+1,maxStages+1)]
            thres_list[-1] = min(thres_list[-1],self.maxIters)
        self.thres_list = thres_list
        self.Nlist = Nlist
        stage_iters = np.r_[thres_list[0], np.diff(thres_list)]
        M = np.sqrt(np.sum([stage_iters[i]/Nlist[i] for i in range(len(Nlist))]))
        eta = sigma*M/D
        self.u = u
        self.D = D
        self.M= M
        self.eta= eta
        self.sigma = sigma
        self.eta_mode = eta_mode
        if self.eta_mode == "duchi":
            self.algo_str += "_duchi"
        totalStages = len(Nlist)
        cur_stage = 0
        cur_N = Nlist[cur_stage]
        cur_thres = thres_list[cur_stage]
        phi_x = phi
        phi_y = phi
        phi_z = phi
        s = np.zeros(self.sample.n)
        theta_old= 1
        theta = 1
        print("algorithm started...")
        start_t = time.time()
        for t in tnrange(self.maxIters+1):
            if t > cur_thres:
                cur_stage += 1
                cur_N = Nlist[cur_stage]
                cur_thres = thres_list[cur_stage]
                print("Iteration",t,":")
                print("N changed to %d"%cur_N)
            grid_mat = self.sample.generate_new_random_feas_grid_mat(cur_N, rng = self.random)
            ut = u*theta_old
            phi_y = (1-theta_old)*phi_x+theta_old*phi_z
            cur = Iterate(phi = phi_y)
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                get_all_from_phi_QP(cur, self.sample.X1, self.sample.Delta, self.w, grid_mat, ut, normalize = self.normalize, njobs= self.njobs )
            cur.time = time.time()-start_t
            s += cur.grad_phi/theta_old
            theta = 2/(1+np.sqrt(1+4/theta_old**2))
            cur.phi_x = phi_x
            cur.phi_y = phi_y
            cur.phi_z = phi_z
            if self.eta_mode == "duchi":
                eta = self.eta * np.sqrt(t+1)
            phi_z = phi - s*theta/(sigma/u+eta)
            phi_x = (1-theta_old)*phi_x+theta_old*phi_z
            theta_old = theta
            self.iterates.append(cur)
            if cur.time > self.maxtime:
                print("algorithm terminated due to time limit\n")
                break
        self.runtime = time.time() - start_t
        print("algorithm finished!\n")
        print("running time: ", "\033[1m",time_to_string(self.runtime),"\033[0;0m","\n" )
        
        self.disc_obj_times = [iterate.time for iterate in self.iterates]
        self.disc_objs = [iterate.obj for iterate in self.iterates]
        self.disc_normalized_objs = [np.mean(iterate.phi)+np.log(iterate.integral)+1 for iterate in self.iterates]
        
        if self.evaluation:
            print("function evaluation started...")
            self.real_objs = []
            self.real_integrals = []
            self.obj_times = []
            self.arg_min = 0
            self.eval_iters=1
            self.min_obj = float('inf')
            for i in tnrange(len(self.iterates)):
                if i % self.eval_iters == 0:
                    phi = self.iterates[i].phi_x
                    obj, integral = self.calcExactFinalObjective(phi,verbose=0)
                    self.real_objs.append(obj)
                    self.real_integrals.append(integral)
                    self.obj_times.append(self.iterates[i].time)
                    if obj < self.min_obj:
                        self.min_obj = obj
                        self.arg_min = i
            print("function evaluation finished!\n\n")


            self.final_phi = self.iterates[self.arg_min].phi
            self.final_time = self.iterates[self.arg_min].time

            print("min obj achieved: ", "\033[1m",self.min_obj,"\033[0;0m")
            if self.plot:
                plt.figure(figsize=(16,9))
                plt.scatter(self.final_time, self.min_obj, c="red",zorder = 1)
                plt.plot(self.obj_times,self.real_objs,color ="orange",zorder=0)
                plt.xlabel("time")
                plt.ylabel("obj")
                plt.show()
    
    def nesterov_smoothing_approx(self,init_method=None,init_kwargs = dict(),D=2,sigma=1,C1=1,Nlist=None,N0list=None,thres_list= None,grid_method="uniform",eta_mode = "constant"):
        self.algo_name = "nesterov_smoothing (approx)"

        u = D/2*C1
        self.C1 = C1
        if N0list is None:
            using_N = True
        else:
            Nlist = [int((N0list[i]**self.sample.d)*self.sample.phat) for i in range(len(N0list))]
            using_N = False
        if thres_list is None:
            maxStages = int(np.ceil(np.log2(self.maxIters)))
            thres_list = [2**i for i in range(maxStages-len(Nlist)+1,maxStages+1)]
            thres_list[-1] = min(thres_list[-1],self.maxIters)
        self.thres_list = thres_list
        self.N0list = N0list
        self.Nlist = Nlist
        if grid_method == "uniform":
            self.algo_str = "nesterov_smoothing_Riemann_%d_%.0e_%.2f_%d_%d_%d"%(D,sigma,C1, self.maxIters, len(N0list),max(N0list))
        elif not using_N:
            self.algo_str = "nesterov_smoothing_rndapprox_%d_%.0e_%.2f_%d_%d_%d"%(D,sigma,C1, self.maxIters, len(N0list),max(N0list))
        else:
            self.algo_str = "nesterov_smoothing_rndapprox_%d_%.0e_%.2f_%d_%d_%d"%(D,sigma,C1, self.maxIters, len(Nlist),max(Nlist))
        self.algo_str += '_Copy%d'%self.random_state
        phi = self.compute_phi_init(init_method,**init_kwargs)
        self.algo_str += self.init_suffix
            
        stage_iters = np.r_[thres_list[0], np.diff(thres_list)]
        M = np.sqrt(np.sum([stage_iters[i]/Nlist[i] for i in range(len(Nlist))]))
        eta = sigma*M/D
        self.u = u
        self.D = D
        self.eta= eta
        self.M= M
        self.sigma = sigma
        self.eta_mode = eta_mode
        if self.eta_mode == "duchi":
            self.algo_str += "_duchi"
        totalStages = len(Nlist)
        cur_stage = 0
        cur_N = Nlist[cur_stage]
        cur_thres = thres_list[cur_stage]
        if using_N:
            self.sample.generate_feas_grid_mat(method=grid_method, N = Nlist[cur_stage],rng=self.random)
        else:
            self.sample.generate_feas_grid_mat(method=grid_method,N0=N0list[cur_stage],rng=self.random)
            cur_N = self.sample.N
            Nlist[cur_stage] = cur_N
        phi_x = phi
        phi_y = phi
        phi_z = phi
        s = np.zeros(self.sample.n)
        theta_old= 1
        theta = 1
        print("algorithm started...")
        start_t = time.time()
        for t in tnrange(self.maxIters+1):
            if t > cur_thres:
                cur_stage += 1
                cur_N = Nlist[cur_stage]
                cur_thres = thres_list[cur_stage]
                print("Iteration",t,":")
                if using_N:
                    print("N changed to %d"%cur_N)
                    self.sample.generate_feas_grid_mat(method=grid_method, N = Nlist[cur_stage],rng=self.random)
                else:
                    print("N0 changed to %d"%N0list[cur_stage])
                    self.sample.generate_feas_grid_mat(method=grid_method,N0=N0list[cur_stage],rng=self.random)
                    cur_N = self.sample.N
                    Nlist[cur_stage] = cur_N
                    print("N changed to %d"%cur_N)
            ut = u*theta_old
            phi_y = (1-theta_old)*phi_x+theta_old*phi_z
            cur = Iterate(phi = phi_y)
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                get_all_from_phi_QP(cur, self.sample.X1, self.sample.Delta, self.w, self.sample.grid_mat, ut, normalize = self.normalize, njobs= self.njobs )
            cur.time = time.time()-start_t
            s += cur.grad_phi/theta_old
            theta = 2/(1+np.sqrt(1+4/theta_old**2))
            cur.phi_x = phi_x
            cur.phi_y = phi_y
            cur.phi_z = phi_z
            if self.eta_mode == "duchi":
                eta = self.eta * np.sqrt(t+1)
            phi_z = phi - s*theta/(sigma/u+eta)
            phi_x = (1-theta_old)*phi_x+theta_old*phi_z
            theta_old = theta
            self.iterates.append(cur)
            if cur.time > self.maxtime:
                print("algorithm terminated due to time limit\n")
                break
        self.Nlist = Nlist
        self.runtime = time.time() - start_t
        print("algorithm finished!\n")
        print("running time: ", "\033[1m",time_to_string(self.runtime),"\033[0;0m","\n" )
        
        self.disc_obj_times = [iterate.time for iterate in self.iterates]
        self.disc_objs = [iterate.obj for iterate in self.iterates]
        self.disc_normalized_objs = [np.mean(iterate.phi)+np.log(iterate.integral)+1 for iterate in self.iterates]
        
        if self.evaluation:
            print("function evaluation started...")
            self.real_objs = []
            self.real_integrals = []
            self.obj_times = []
            self.arg_min = 0
            self.eval_iters=1
            self.min_obj = float('inf')
            for i in tnrange(len(self.iterates)):
                if i % self.eval_iters == 0:
                    phi = self.iterates[i].phi_x
                    obj, integral = self.calcExactFinalObjective(phi,verbose=0)
                    self.real_objs.append(obj)
                    self.real_integrals.append(integral)
                    self.obj_times.append(self.iterates[i].time)
                    if obj < self.min_obj:
                        self.min_obj = obj
                        self.arg_min = i
            print("function evaluation finished!\n\n")


            self.final_phi = self.iterates[self.arg_min].phi
            self.final_time = self.iterates[self.arg_min].time

            print("min obj achieved: ", "\033[1m",self.min_obj,"\033[0;0m")
            if self.plot:
                plt.figure(figsize=(16,9))
                plt.scatter(self.final_time, self.min_obj, c="red",zorder = 1)
                plt.plot(self.obj_times,self.real_objs,color ="orange",zorder=0)
                plt.xlabel("time")
                plt.ylabel("obj")
                plt.show()        

    def calcExactFinalObjective(self,phi,verbose=0):
        integral = self.sample.calcExactIntegral(phi,verbose=verbose)
        return phi@self.w + np.log(integral) + 1, integral
    
    def to_dict(self,contents = "all"):
        d = dict()
        d2 = dict()
        if contents in ("all","info"):
            for key in self.__dict__:
                if key not in ['sample','iterates','random','solver_class','solver','ws_duals','solver_method','lbmodel','phi','v','rad','rad_constr','ub_constrs','lb_constrs','int_pos','status','stepModel']:
                    if key not in ["final_phi","w"]:
                        d[key] = getattr(self,key)
                    else:
                        d[key] = getattr(self,key).tolist()
            d['n'] = self.sample.n
            d['d'] = self.sample.d
        if contents in ("all","hist"):
            d2['phi_hist'] = [it.phi.tolist() for it in self.iterates]
            d2['grad_phi_hist'] = [it.grad_phi.tolist() for it in self.iterates]
        
        if contents =="info":
            return d
        elif contents =="hist":
            return d2
        elif contents=="all":
            d.update(d2)
            return d
            
    def to_json(self,path):
        d = dict()
        d2 = dict()
        for key in self.__dict__:
            if key not in ['sample','iterates','random','solver_class','solver','ws_duals','solver_method','lbmodel','phi','v','rad','rad_constr','ub_constrs','lb_constrs','int_pos','status','stepModel']:
                if key not in ["final_phi","w"]:
                    d[key] = getattr(self,key)
                else:
                    d[key] = getattr(self,key).tolist()
        
        d['n'] = self.sample.n
        d['d'] = self.sample.d
        d2['phi_hist'] = [it.phi.tolist() for it in self.iterates]
        d2['grad_phi_hist'] = [it.grad_phi.tolist() for it in self.iterates]
        with open(path+"info/%s_%s_info.json"%(self.sample.sample_str,self.algo_str),'w') as f:
            json.dump(d,f)
        with open(path+"hist/%s_%s_hist.json"%(self.sample.sample_str,self.algo_str),'w') as f:
            json.dump(d2,f)
        with open(path+"soln/%s_%s_soln.npy"%(self.sample.sample_str,self.algo_str),'wb') as f:
            np.save(f, self.final_phi)            

            
            
            