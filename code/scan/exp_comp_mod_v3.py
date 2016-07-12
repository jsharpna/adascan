from adascan import *
import numpy as np
from scipy import signal
import scipy.stats as stats
import pickle
import time

l = 8
n = 2**l
h_low = 6
sigma = 1.
d = 2
h = np.array([18,15])
mu = 6.
trials = 25
ada_null = []
ada_alt = []
mult_null = []
mult_alt = []
oracle_null = []
oracle_alt = []
mod_null = []
mod_alt = []
for t in range(trials):
    x,loc = make_rect(n,d,h)
    eps = make_square_noise(n,d,1.)
    y = mu / (h[0]*h[1])**0.5 * x + eps
    oracle_alt.append(oracle_scan_2d(y,h)[1])
    oracle_null.append(oracle_scan_2d(eps,h)[1])
    mult_alt.append(np.min(mult_scan_ts_2d(y,h_low)[1]))
    mult_null.append(np.min(mult_scan_ts_2d(eps,h_low)[1]))
    ada_alt.append(np.min(adascan_ts_2d(y,h_low)[1]))
    ada_null.append(np.min(adascan_ts_2d(eps,h_low)[1]))
    mod_alt.append(np.min(adascan_mod_2d(y,h_low)[1]))
    mod_null.append(np.min(adascan_mod_2d(eps,h_low)[1]))

filename = "sim_ROC_v3_" + str(time.time()) + ".dat"
sfile = open(filename,"wb")
pickle.dump((oracle_null,oracle_alt,mult_null,mult_alt,ada_null,ada_alt,mod_null,mod_alt), sfile)
sfile.close()

