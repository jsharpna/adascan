from adascan import *
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib import cm
import pylab
import scipy.stats as stats
import os
import pickle

ii=2
ROC_files = os.listdir("../datav2")
#sim_comp = [f for f in ROC_files if "ROC_v1" in f]
sim_comp = [f for f in ROC_files if "ROC_v" + str(ii) + "_" in f]

def plot_ROC(null,alt,linec,lw=2):
    ROC = [0.]
    null_sort = np.sort(null)
    for i in range(len(null)):
        ROC.append(1.-np.mean(alt > null_sort[i]))
    plt.plot(np.arange(len(null)+1)/float(len(null)),ROC,linec,linewidth=lw)

oracle_null,oracle_alt,mult_null,mult_alt,ada_null,ada_alt,mod_null,mod_alt = [], [], [], [], [], [], [], []

for f in sim_comp:
    oracle_nullt,oracle_altt,mult_nullt,mult_altt,ada_nullt,ada_altt,mod_nullt,mod_altt = pickle.load(open("../datav2/" + f,"r"))
    oracle_null = oracle_null + oracle_nullt
    mult_null = mult_null + mult_nullt
    ada_null = ada_null + ada_nullt
    mod_null = mod_null + mod_nullt
    oracle_alt = oracle_alt + oracle_altt
    mult_alt = mult_alt + mult_altt
    ada_alt = ada_alt + ada_altt
    mod_alt = mod_alt + mod_altt

lty = ['b-','g-.','r--','m:']
lw = [2,3,2,3.5]

fs = 18
plt.figure(num=None, figsize=(5.15, 5), dpi=80, facecolor='w', edgecolor='k')    
plot_ROC(oracle_null,oracle_alt,lty[0],lw[0])
plot_ROC(mult_null,mult_alt,lty[1],lw[1])
plot_ROC(ada_null,ada_alt,lty[2],lw[2])
plot_ROC(mod_null,mod_alt,lty[3],lw[3])

plt.xlim([-0.005,1])
plt.ylim([0,1.005])
plt.ylabel('true discovery rate',fontsize=fs)
plt.xlabel('false discovery rate',fontsize=fs)
plt.legend(('oracle','multiscale','adaptive','modified'),loc='lower right',fontsize=fs)
plt.savefig('ROC_v' + str(ii) + '_mod.png', bbox_inches='tight')
#plt.show()

fs = 18
plt.figure(num=None, figsize=(5.15, 5), dpi=80, facecolor='w', edgecolor='k')    
plot_ROC(oracle_null,oracle_alt,lty[0],lw[0])
plot_ROC(mult_null,mult_alt,lty[1],lw[1])
plot_ROC(ada_null,ada_alt,lty[2],lw[2])

plt.xlim([-0.005,1])
plt.ylim([0,1.005])
plt.ylabel('true discovery rate',fontsize=fs)
plt.xlabel('false discovery rate',fontsize=fs)
#plt.legend(('oracle','multiscale','adaptive'),loc='lower right',fontsize=fs)
plt.savefig('ROC_v' + str(ii) + '_noleg.png', bbox_inches='tight')
#plt.show()

fs = 18
plt.figure(num=None, figsize=(5.15, 5), dpi=80, facecolor='w', edgecolor='k')    
plot_ROC(oracle_null,oracle_alt,lty[0],lw[0])
plot_ROC(mult_null,mult_alt,lty[1],lw[1])
plot_ROC(ada_null,ada_alt,lty[2],lw[2])
plot_ROC(mod_null,mod_alt,lty[3],lw[3])

plt.xlim([-0.005,1])
plt.ylim([0,1.005])
plt.ylabel('true discovery rate',fontsize=fs)
plt.xlabel('false discovery rate',fontsize=fs)
#plt.legend(('oracle','multiscale','adaptive','modified'),loc='lower right',fontsize=fs)
plt.savefig('ROC_v' + str(ii) + '_mod_noleg.png', bbox_inches='tight')
#plt.show()

fs = 18
plt.figure(num=None, figsize=(5.15, 5), dpi=80, facecolor='w', edgecolor='k')    
plot_ROC(oracle_null,oracle_alt,lty[0],lw[0])
plot_ROC(mult_null,mult_alt,lty[1],lw[1])
plot_ROC(ada_null,ada_alt,lty[2],lw[2])

plt.xlim([-0.005,1])
plt.ylim([0,1.005])
plt.ylabel('true discovery rate',fontsize=fs)
plt.xlabel('false discovery rate',fontsize=fs)
#plt.legend(('oracle','multiscale','adaptive'),loc='lower right',fontsize=fs)
plt.savefig('ROC_v' + str(ii) + '_noleg.png', bbox_inches='tight')
#plt.show()



ROC_files = os.listdir("../datav2")
ii = 9
sim_comp = [f for f in ROC_files if "v"+str(ii) in f]

oracle_null = []
mult_null = []
ada_null = []
for f in sim_comp:
    oracle_t, mult_t, ada_t, run_t = pickle.load(open("../datav2/" + f,"r"))
    oracle_null = oracle_null + oracle_t
    mult_null = mult_null + mult_t
    ada_null = ada_null + ada_t
    
T = len(oracle_null)
fs = 18
plt.figure(num=None, figsize=(5.15, 5), dpi=80, facecolor='w', edgecolor='k')
plt.xlim([-0.005,1])
plt.ylim([0,1.005])
plt.plot(np.arange(T) / float(T), np.sort(oracle_null),lty[0],linewidth=lw[0],label="oracle")
plt.plot(np.arange(T) / float(T), np.sort(mult_null),lty[1],linewidth=lw[1],label="multiscale")
plt.plot(np.arange(T) / float(T), np.sort(ada_null),lty[2],linewidth=lw[2],label="adaptive")
plt.plot([0,1],[0,1],'k')
plt.xlabel('unif(0,1)',fontsize=fs)
plt.ylabel("sorted P-values",fontsize=fs)
plt.savefig('QQ_v' + str(ii) + '_noleg.png', bbox_inches='tight')
plt.show()

plt.figure(num=None, figsize=(5.15, 5), dpi=80, facecolor='w', edgecolor='k')
plt.xlim([-0.005,1])
plt.ylim([0,1.005])
plt.plot(np.arange(T) / float(T), np.sort(oracle_null),lty[0],linewidth=lw[0],label="oracle")
plt.plot(np.arange(T) / float(T), np.sort(mult_null),lty[1],linewidth=lw[1],label="multiscale")
plt.plot(np.arange(T) / float(T), np.sort(ada_null),lty[2],linewidth=lw[2],label="adaptive")
plt.plot([0,1],[0,1],'k')
plt.xlabel('unif(0,1)',fontsize=fs)
plt.ylabel("sorted P-values",fontsize=fs)
plt.legend(loc = 4,fontsize=fs)
plt.savefig('QQ_v' + str(ii) + '.png', bbox_inches='tight')
plt.show()




ROC_files = os.listdir("../datav2")
sim_comp = [f for f in ROC_files if "ROC_v5" in f]
#sim_comp = [f for f in ROC_files if "ROC_v6" in f]

E = 6
DS = []
for f in sim_comp:
    DS.append(pickle.load(open("../datav2/" + f,"r")))

s1e = [[] for e in range(E)]
s0e = [[] for e in range(E)]
stimes = [[] for e in range(E)]    
for e in range(E):
    for ds in DS:
        s0e[e] = s0e[e] + ds[0][e]
        s1e[e] = s1e[e] + ds[1][e]
        stimes[e] = stimes[e] + ds[2][e]
        
def plot_ROC(null,alt,label,lty,lw):
    ROC = []
    null_sort = np.sort(null)
    for i in range(len(null)):
        ROC.append(1.-np.mean(alt > null_sort[i]))
    plt.plot(np.arange(len(null))/float(len(null)),ROC,lty,linewidth=lw,label=label)

fs = 18
ialpha_grid = (np.arange(E) + 1.)
lnums = [0,1,2,4]
#lnums = [0,1,3,5]
eps_grid = (8. / ialpha_grid)**0.5
lty = ['b-','g-.','r--','m:']
lw = [2,3,2,3.5]
plt.figure(num=None, figsize=(5.15, 5), dpi=80, facecolor='w', edgecolor='k')    
plt.xlim([-0.005,1])
plt.ylim([0,1.005])
for i in range(4):
    e = lnums[i]
    if i == 0:
        la = "$\epsilon=" + str(int(eps_grid[e]*100)/100.) + "$"
    else:
        la = "$" + str(int(eps_grid[e]*100)/100.) + "$"
    plot_ROC(s0e[e],s1e[e],la,lty[i],lw[i])

plt.ylabel('true discovery rate',fontsize=fs)
plt.xlabel('false discovery rate',fontsize=fs)
plt.legend(loc='lower right',fontsize=fs)
plt.savefig('ROC_v5.png', bbox_inches='tight')

plt.figure(num=None, figsize=(5.15, 5), dpi=80, facecolor='w', edgecolor='k')    
plt.xlim([-0.005,1])
plt.ylim([0,1.005])
for i in range(4):
    e = lnums[i]
    if i == 0:
        la = "$\epsilon=" + str(int(eps_grid[e]*100)/100.) + "$"
    else:
        la = "$" + str(int(eps_grid[e]*100)/100.) + "$"
    plot_ROC(s0e[e],s1e[e],la,lty[i],lw[i])

plt.ylabel('true discovery rate',fontsize=fs)
plt.xlabel('false discovery rate',fontsize=fs)
plt.savefig('ROC_v5_noleg.png', bbox_inches='tight')

#plt.savefig('ROC_v6.png', bbox_inches='tight')
#plt.show()

T = len(stimes[0])
i0 = int(T*.05)
i1 = int(T*.95)
tmeans = np.array([np.mean(stimes[i]) for i in range(E)])
tlow   = [np.min(stimes[i]) for i in range(E)]
tup    = [np.max(stimes[i]) for i in range(E)]
tlq   = np.array([np.sort(stimes[i])[i0] for i in range(E)])
tuq    = np.array([np.sort(stimes[i])[i1] for i in range(E)])
plt.figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k')
plt.plot(eps_grid,tmeans,'k-')
for i in range(E):
    plt.plot([eps_grid[i]]*2,[tlow[i],tup[i]],'k-',marker='_')
plt.plot(eps_grid,tlq-4.*.15,linestyle='none',marker='^',markersize=10)
plt.plot(eps_grid,tuq+4.*.25,linestyle='none',marker='v',markersize=10)
plt.xlabel("$\epsilon$ parameter in Alg.1",fontsize=fs)
plt.ylabel("Running time (sec)",fontsize=fs)
plt.savefig('run_time_v6.png', bbox_inches='tight')
#plt.show()



