from adascan import *
import numpy as np
from scipy import signal
import scipy.stats as stats
import pickle
import time

l = 9
n = 2**l
h_low = 32
d = 2
h = np.array([32,32])
sigma = 1.
d = 2
trials = 25
ada_null = []
mult_null = []
oracle_null = []
mod_null = []
run_times = []
for t in range(trials):
    eps = make_square_noise(n,d,1.)
    oracle_null.append(oracle_scan_2d(eps,h)[1])
    mult_null.append(np.min(mult_scan_2d(eps,h_low)[1]))
    t0 = time.time()
    ada_null.append(np.min(adascan_2d(eps,h_low)[1]))
    run_times.append(time.time() - t0)

filename = "sim_null_v10_" + str(int(time.time()*100)) + ".dat"
sfile = open(filename,"wb")
pickle.dump((oracle_null,mult_null,ada_null,run_times), sfile)
sfile.close()
