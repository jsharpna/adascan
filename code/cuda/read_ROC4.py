import pickle
from matplotlib import pyplot as plt
import numpy as np

def plot_ROC(null,alt,linec,lw=2,lab=None):
    ROC = [0.]
    null_sort = np.sort(-null)
    for i in range(len(null)):
        ROC.append(1.-np.mean(-alt > null_sort[i]))
    plt.plot(np.arange(len(null)+1)/float(len(null)),ROC,linec,linewidth=lw,label=lab)


ofs = ["stats4-733838.pi","stats4-736124.pi","stats4-736666.pi","stats4-837664.pi","stats4-736222.pi","stats4-739459.pi","stats4-938287.pi"]

null,alt = pickle.load(open('stats4-735235.pi','r'))
for j in range(7):
    nullt, altt = pickle.load(open(ofs[j],'r'))
    null = np.vstack((null,nullt))
    alt = np.vstack((alt,altt))

leg = True

i=4
lty = ['b-','g-.','r--','m:']
lw = [2,3,2,3.5]
fs=11
plt.figure(num=None, figsize=(5.15, 5), dpi=80, facecolor='w', edgecolor='k')    

if leg:
    plot_ROC(null[:,0],alt[:,0],lty[2],lw=lw[2],lab="Adaptive") #Adascan - red
    plot_ROC(null[:,2],alt[:,2],lty[1],lw=lw[1],lab="Multiscale") #Multiscale - green
    plot_ROC(null[:,3],alt[:,3],lty[0],lw=lw[0],lab="Oracle") #Oracle - blue
else:    
    plot_ROC(null[:,0],alt[:,0],lty[2],lw=lw[2]) #Adascan - red
    plot_ROC(null[:,2],alt[:,2],lty[1],lw=lw[1]) #Multiscale - green
    plot_ROC(null[:,3],alt[:,3],lty[0],lw=lw[0]) #Oracle - blue
plt.xlim([-0.005,1])
plt.ylim([0,1.005])
plt.ylabel('true discovery rate',fontsize=fs)
plt.xlabel('false discovery rate',fontsize=fs)
plt.savefig('ROC' + str(i) + '_nolab.png', bbox_inches='tight')
plt.legend(loc='lower right',fontsize=fs)
plt.savefig('ROC' + str(i) + '.png', bbox_inches='tight')
plt.show()

j = 0
kappa_th = -np.log(4**2. * (2.*np.pi)**0.5)
kappa_mod = np.mean(null[:,j]) - 0.5772
alpha_th = 1. - np.exp(-np.exp(-(null[:,j] - kappa_th)))
alpha_th = np.sort(alpha_th)
alpha_mod = 1. - np.exp(-np.exp(-(null[:,j] - kappa_mod)))
alpha_mod = np.sort(alpha_mod)
T = len(null)
fs = 10
plt.figure(num=None, figsize=(5.15, 5), dpi=80, facecolor='w', edgecolor='k')
plt.xlim([-0.005,1])
plt.ylim([0,1.005])
plt.plot(np.arange(T) / float(T), np.sort(alphas),linewidth=2,label="oracle")
plt.plot([0,1],[0,1],'k')
plt.xlabel('unif(0,1)',fontsize=fs)
plt.ylabel("sorted P-values",fontsize=fs)
plt.legend(loc = 4,fontsize=fs)
plt.savefig('QQ' + str(i) + '.png', bbox_inches='tight')
plt.show()


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



