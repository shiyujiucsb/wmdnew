#start
import pdb, sys, numpy as np, pickle, multiprocessing as mp

load_file = sys.argv[1]
save_file = sys.argv[2]

with open(load_file) as f:
    [X, BOW_X, y, C, words] = pickle.load(f)
n = np.shape(X)
n = n[0]
D = np.zeros((n,n))
for i in xrange(n):
    bow_i = BOW_X[i]
    bow_i = bow_i / np.sum(bow_i)
    bow_i = bow_i.tolist()
    BOW_X[i] = bow_i
    X_i = X[i].T
    X_i = X_i.tolist()
    X[i] = X_i

nBuckets = 20
nDim = len(X[0][0])
import random as r
r.seed(10)
LSH = []
for i in range(nBuckets):
    vec = []
    for j in range(nDim):
        vec.append(r.uniform(-1,1))
    LSH.append(vec)

def dot(v, u):
    res = 0.0
    for i in range(len(v)):
        res += v[i] * u[i]
    return res

def sign(V):
    sig = ""
    for i in range(len(LSH)):
        if dot(V, LSH[i]) > 0:
            sig+="1"
        else:
            sig+="0"
    return int(sig, 2)

def helper(X1, BOW1, X2, BOW2):
    slots1 = {}
    slots2 = {}
    for i in range(len(X1)):
        s = sign(X1[i])
        if s in slots1:
            slots1[s] += BOW1[i]
        else:
            slots1[s] = BOW1[i]
    for i in range(len(X2)):
        s = sign(X2[i])
        if s in slots2:
            slots2[s] += BOW2[i]
        else:
            slots2[s] = BOW2[i]
    res = 0.0
    for k in slots1.keys():
        if k in slots2:
            res += slots1[k] * slots2[k]
    return res

def get_wmd(ix):
    n = np.shape(X)
    n = n[0]
    Di = np.zeros((1,n))
    i = ix
    #print '%d out of %d' % (i, n)
    for j in xrange(i):
        #Di[0,j] = emd( (X[i], BOW_X[i]), (X[j], BOW_X[j]), distance)
        Di[0,j] = helper(X[i], BOW_X[i], X[j], BOW_X[j])
        #if Di[0,j]>0.4 and min(len(X[i]), len(X[j])) >10:
        print(ix," and ",j,": ", Di[0, j])
    return Di


def main():
    n = np.shape(X)
    n = n[0]
    pool = mp.Pool(processes=8)

    pool_outputs = pool.map(get_wmd, list(range(n)))
    pool.close()
    pool.join()

    WMD_D = np.zeros((n,n))
    for i in xrange(n):
        WMD_D[:,i] = pool_outputs[i]

    with open(save_file, 'w') as f:
        pickle.dump(WMD_D, f)

if __name__ == "__main__":
    main()
