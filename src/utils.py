import os
import numpy as np
import numba as nb

def create_folder(storage_path):
    if not os.path.isdir(storage_path):
        os.makedirs(storage_path,exist_ok=True)
    lsdir =  os.listdir(storage_path)
    for item in ["info","hist","soln","figs"]:
        if item not in lsdir:
            os.makedirs(storage_path+item+"/",exist_ok=True)
        if item == "figs":
            lsdir_figs = os.listdir(storage_path+item+"/")
            for item1 in ["crop","raw"]:
                if item1 not in lsdir_figs:
                    os.makedirs(storage_path+item+"/"+item1+"/",exist_ok=True)
                    
                    
def time_to_string(runtime):
    seconds = runtime%60
    runmins = (runtime-seconds)/60
    mins = int(runmins%60)
    runhrs = (runmins-mins)/60
    hrs = int(runhrs)
    return "%.2d:%.2d:%05.2f"%(hrs,mins,seconds)

def multivariate_laplace(n,d,rng=None, random_state=None):
    rng = rng if rng is not None else np.random.RandomState(random_state)
    X = rng.randn(n,d)
    Z = rng.exponential(size=(n,1))
    return X*np.sqrt(Z)


@nb.njit(cache=True)
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@nb.njit(cache=True)
def np_apply_along_axis_kd(funckd, axis, arr, k = -1):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        k = k if k > 0 else arr.shape[0]
        result = np.empty((k,arr.shape[1]))
        for i in range(arr.shape[1]):
            result[:, i] = funckd(arr[:, i])
    else:
        k = k if k > 0 else arr.shape[1]
        result = np.empty((arr.shape[0],k))
        for i in range(arr.shape[0]):
            result[i, :] = funckd(arr[i, :])
    return result

@nb.njit(cache=True)
def split(n, B):
    sep = n//B
    rem = n%B
    indices = []
    last = 0
    cur = 0
    for i in range(B):
        cur = last + sep + (i < rem)
        indices.append(cur)
        last = cur
    return indices

