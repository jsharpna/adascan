import numpy as np
import theano
from theano.sandbox.cuda.dnn import dnn_pool, dnn_conv
from itertools import izip, product

def build_filters(h_low,h_high,verbose = False):
    H = np.arange(h_low,h_high)
    kern = np.zeros(((h_high - h_low)**2,1,h_high,h_high),dtype=np.float32)
    t = 0
    for h1,h2 in product(H,H):
        kern[t,0,0:h1,0:h2] = 1./(h1*h2)
        t += 1
    return kern

def dyad_2d(y,down_x,down_y,L1,L2):
    _,_,n1,n2 = y.shape
    dyad = {}
    for i1 in range(L1):
        for i2 in range(L2):
            N1 = 2**(L1 - i1)
            N2 = 2**(L2 - i2)
            if i1 == 0 and i2 == 0:
                dyad[i1*L1 + i2] = y*2.
            elif i2 == 0:
                dyad[i1*L1 + i2] = down_x(dyad[(i1-1)*L1 + i2])
            else:
                dyad[i1*L1 + i2] = down_y(dyad[i1*L1 + (i2 - 1)])
    return dyad

def stat_calc(scores,N,method='adascan_mod',h_low = 1):
    if method=='adascan_mod':
        nu = (2.*(np.log(N/scores[:,0]) + np.log(N/scores[:,1])))**0.5
    if method=='adascan':
        nu = (2.*(np.log(N/scores[:,0] * (1 + scores[:,0] / h_low)**2.) + np.log(N/scores[:,1] * (1 + scores[:,1] / h_low)**2.)))**0.5
    if method=='oracle':
        nu = (2.*(np.log(N/scores[:,0]) + np.log(N/scores[:,1])))**0.5
    if method=='multi':
        h_low = scores[0,0]
        nu = (2.*(np.log(N/h_low) + np.log(N/h_low)))**0.5
    stat = np.max((scores[:,2] - nu)*nu - 7.*np.log(nu))
    return stat

def generate_image_null(N):
    x = np.asarray(np.random.normal(0,1,N**2),dtype=np.float32)
    x.shape = (1,1,N,N)
    return x

def generate_image_alt(N,mu,H):
    x = np.asarray(np.random.normal(0,1,N**2),dtype=np.float32)
    x.shape = (1,1,N,N)
    loc1, loc2 = np.random.randint(0,N-H[0]), np.random.randint(0,N-H[1])
    x[0,0,loc1:(loc1+H[0]),loc2:(loc2+H[1])] =  x[0,0,loc1:(loc1+H[0]),loc2:(loc2+H[1])] + mu
    return x

class EpsScan:

    def __init__(self,h_low,h_high,method="adascan_mod"):
        kern = build_filters(h_low,h_high)    
        sharedKern = theano.shared(kern,name='sharedKern')
        input = theano.tensor.tensor4(name='input')
        self.conv_fun   = theano.function([input],dnn_conv(input,sharedKern))
        self.down_x = theano.function([input],dnn_pool(input,(2,1),stride=(2,1),mode='average_inc_pad'))
        self.down_y = theano.function([input],dnn_pool(input,(1,2),stride=(1,2),mode='average_inc_pad'))
        self.h_low, self.h_high, self.method = h_low, h_high, method
        
    def build_dyad(self,x,L):
        self.dyad = dyad_2d(x,self.down_x,self.down_y,L,L)
        self.L = L
        self.N = x.shape[2]

    def build_scores(self,L):
        if not self.dyad:
            print("run build_dyad")
        else:
            num_sc = self.L**2 * (self.h_high - self.h_low)**2
            scores = np.zeros((num_sc,3))
            t_lar = 0
            for i1,i2 in product(range(self.L),range(self.L)):
                dl = i1*self.L + i2
                conv_temp = self.conv_fun(self.dyad[dl])
                t = 0
                for h1, h2 in product(range(self.h_low,self.h_high),range(self.h_low,self.h_high)):
                    mcur = np.max(conv_temp[0,t,:,:])
                    H1, H2 = h1*2**i1, h2*2**i2
                    scores[t_lar,:] = [H1,H2,mcur*(H1*H2)**0.5 / 2.]
                    t += 1
                    t_lar += 1
            self.scores = scores
            
