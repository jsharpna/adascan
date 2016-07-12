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
h = np.array([82,21])

mu = 20.
E = 6
s1a, s0a, s1m, s0m, stime = [], [], [], [], []
ialpha = 5
eps = (8. / ialpha)**0.5
trials = 5
for t in range(trials):
    x,loc = make_rect(n,d,h)
    noise = make_square_noise(n,d,1.)
    y = mu / (h[0]*h[1])**0.5 * x + noise
    dyad = dyad_2d(y)
    s1a.append(np.min(eps_adascan_2d(y,eps=eps,h_low=h_low,dyad=dyad)[1]))
    t0 = time.time()
    s0a.append(np.min(eps_adascan_2d(noise,eps=eps,h_low=h_low,dyad=dyad)[1]))
    stime.append(time.time() - t0)
    s1m.append(np.max(eps_scan_2d(y,eps=eps,h_low=h_low,dyad=dyad)[1]))
    s0m.append(np.max(eps_scan_2d(noise,eps=eps,h_low=h_low,dyad=dyad)[1]))
        
filename = "epsada_ROC_v11_" + str(int(time.time()*100)) + ".dat"
sf = open(filename,"wb")
pickle.dump((s0e,s1e,stime), sf)
sf.close()

