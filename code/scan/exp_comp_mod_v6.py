from adascan import *
import numpy as np
from scipy import signal
import scipy.stats as stats
import pickle
import time

l = 9
n = 2**l
y_sig = np.zeros((n,n))
d = 2
h_low = 12
h = np.array([82,35])

mu = 5.
E = 6
s1e = [[] for i in range(E)]
s0e = [[] for i in range(E)]
stime = [[] for i in range(E)]
ialpha_grid = (np.arange(E) + 1.)
eps_grid = (8. / ialpha_grid)**0.5
trials = 2
for t in range(trials):
    x,loc = make_rect(n,d,h)
    noise = make_square_noise(n,d,1.)
    y = mu / (h[0]*h[1])**0.5 * x + noise
    for e in range(E):
        eps = eps_grid[e]
        s1e[e].append(np.min(eps_adascan_2d(y,eps=eps,h_low=h_low)[1]))
        t0 = time.time()
        s0e[e].append(np.min(eps_adascan_2d(noise,eps=eps,h_low=h_low)[1]))
        stime[e].append(time.time() - t0)

filename = "epsada_ROC_v6_" + str(int(time.time())) + ".dat"
sf = open(filename,"wb")
pickle.dump((s0e,s1e,stime), sf)
sf.close()

