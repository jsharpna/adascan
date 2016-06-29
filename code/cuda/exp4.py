from gpuadascan import *
import pickle

N = 2**11
s = 4
h_up = 2**9
N_large = N * s - h_up * (s - 1)
es = EpsScan(10,20)
T = 500
H = [148,431]
mu = 6./(H[0]*H[1])**0.5

def run(hypo = 'null'):
    if hypo == 'alt':
        X = generate_image_alt(N_large,mu,H)
    if hypo == 'null':
        X = generate_image_null(N_large)
    nsmod, nsada, nsmul, nsora = (-np.inf,-np.inf,-np.inf,-np.inf)
    for i in range(s):
        for j in range(s):
            x = X[:,:,i*(N - h_up):(i*(N - h_up) + N), j*(N - h_up):(j*(N - h_up) + N)]
            es.build_dyad(x,6)
            es.build_scores(6)
            nsmod = max(nsmod,stat_calc(es.scores,N,method='adascan_mod',h_low = 10))
            nsada = max(nsada,stat_calc(es.scores,N,method='adascan',h_low = 10))
            nsmul = max(nsmul,stat_calc(es.scores,N,method='multi',h_low = 10))
            orac_sc = np.argmin(np.sum(np.abs(es.scores[:,0:2] - np.array(H)),axis=1))
            nsora = max(nsora,stat_calc(es.scores[orac_sc:(orac_sc+1),:],N,method='oracle',h_low = 10))                
            es.build_dyad(x,6)
    return(np.array([nsada,nsmod,nsmul,nsora]))


null_stats = np.zeros((T,4))
alt_stats = np.zeros((T,4))
for t in range(T):
    null_stats[t,:] = run('null')
    alt_stats[t,:] = run('alt')


pickle.dump((null_stats,alt_stats),open('stats4.pi','wb'))

