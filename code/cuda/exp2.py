from gpuadascan import *
import pickle

N = 2**11
es = EpsScan(10,20)
T = 10
H = [148,231]
mu = 5./(H[0]*H[1])**0.5

def run(hypo = 'null'):
    if hypo == 'null':
        x = generate_image_null(N)
        es.build_dyad(x,6)
        es.build_scores(6)
        nsmod = stat_calc(es.scores,N,method='adascan_mod',h_low = 10)
        nsada = stat_calc(es.scores,N,method='adascan',h_low = 10)
        nsmul = stat_calc(es.scores,N,method='multi',h_low = 10)
        orac_sc = np.argmin(np.sum(np.abs(es.scores[:,0:2] - np.array(H)),axis=1))
        nsora = stat_calc(es.scores[orac_sc:(orac_sc+1),:],N,method='oracle',h_low = 10)        
    if hypo == 'alt':
        x = generate_image_alt(N,mu,H)
        es.build_dyad(x,6)
        es.build_scores(6)
        nsmod = stat_calc(es.scores,N,method='adascan_mod',h_low = 10)
        nsada = stat_calc(es.scores,N,method='adascan',h_low = 10)
        nsmul = stat_calc(es.scores,N,method='multi',h_low = 10)
        orac_sc = np.argmin(np.sum(np.abs(es.scores[:,0:2] - np.array(H)),axis=1))
        nsora = stat_calc(es.scores[orac_sc:(orac_sc+1),:],N,method='oracle',h_low = 10)        
    return(np.array([nsada,nsmod,nsmul,nsora]))


null_stats = np.zeros((T,4))
alt_stats = np.zeros((T,4))
for t in range(T):
    null_stats[t,:] = run('null')
    alt_stats[t,:] = run('alt')


pickle.dump((null_stats,alt_stats),open('stats11.pi','wb'))

