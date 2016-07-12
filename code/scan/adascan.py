import numpy as np
from scipy import signal

def make_square(n,d,k):
    loc = []
    for i in range(d):
        loci = np.random.randint(0,n - k - 1)
        loc.append(loci)
        newl = np.zeros(n)
        newl[loci:(loci+k)] = 1.
        try:
            x = np.outer(x,newl)
        except NameError:
            x = newl    
    return x,loc

def make_rect(n,d,r):    
    assert len(r) == d
    loc = []
    for i in range(d):
        loci = np.random.randint(0,n - r[i] - 1)
        loc.append(loci)
        newl = np.zeros(n)
        newl[loci:(loci+r[i])] = 1.
        try:
            x = np.outer(x,newl)
        except NameError:
            x = newl    
    return x,loc

def make_square_noise(n,d,sdv=1.):
    Xshape = ()
    for i in range(d):
        Xshape += (n,)
    eps = np.random.normal(0.,1.,n**d)
    eps.shape = Xshape
    return eps * sdv

def conv_square(y,d,r):
    for i in range(d):
        try:
            square = np.outer(square,np.ones(r))
        except NameError:
            square = np.ones(r)    
    return signal.fftconvolve(y,square)

def conv_rect(y,d,r):
    for i in range(d):
        try:
            rect = np.outer(rect,np.ones(r[i]))
        except NameError:
            rect = np.ones(r[i])
    return signal.fftconvolve(y,rect,mode='valid')

def conv_rect2(y,d,r):
    if r[0]==1 and r[1] == 1:
        return y
    C = np.zeros(y.shape)
    for i1 in range(y.shape[0] - r[0]):
        for i2 in range(y.shape[1] - r[1]):
            C[i1,i2] = np.sum(y[i1:(i1+r[0]),i2:(i2+r[1])])
    return C

def mult_scan_pvalue(n,d,h_low,s):
    v = (2. * d * np.log( (n / h_low) ))**0.5
    kappa = - np.log(4.**d * (2.*np.pi)**0.5)
    tau = (s - v)*v - (kappa + (4.*d - 1.)*np.log(v))
    alpha = 1 - np.exp(-np.exp(-tau))
    return alpha

def mult_scan_ts(n,d,h_low,s):
    v = (2. * d * np.log( (n / h_low) ))**0.5
    kappa = - np.log(4.**d * (2.*np.pi)**0.5)
    tau = (s - v)*v - (kappa + (4.*d - 1.)*np.log(v))
    alpha = np.exp(-tau)
    return tau

def oracle_scan_pvalue(n,h,s):
    d = len(h)
    v = (2. * np.sum( np.log( (n / h) )))**0.5
    kappa = - np.log((2.*np.pi)**0.5)
    tau = (s - v)*v - (kappa + (2.*d - 1.)*np.log(v))
    alpha = 1 - np.exp(-np.exp(-tau))
    return alpha

def oracle_scan_2d(y,h):
    n = y.shape[0]
    Z = np.max(conv_rect(y,2,h)) / (h[0]*h[1])**0.5
    alpha = oracle_scan_pvalue(n,h,Z)
    return 0, alpha

def mult_scan_2d(y,h_low = 1.):
    n = y.shape[0]
    max_c = []
    L = int(n/(np.e*h_low))
    for r1 in np.arange(L) + h_low:
        for r2 in np.arange(L) + h_low:
            h = np.array([r1,r2])
            Z = np.max(conv_rect(y,2,h)) / (r1*r2)**0.5
            max_c.append(mult_scan_pvalue(n,2,h_low,Z))
    return 0, max_c

def mult_scan_ts_2d(y,h_low = 1.):
    n = y.shape[0]
    max_c = []
    L = int(n/(np.e*h_low))
    for r1 in np.arange(L) + h_low:
        for r2 in np.arange(L) + h_low:
            h = np.array([r1,r2])
            Z = np.max(conv_rect(y,2,h)) / (r1*r2)**0.5
            max_c.append(mult_scan_ts(n,2,h_low,Z))
    return 0, max_c

def adascan_thresh(n,h,h_low,tau):
    d = len(h)
    v = (2. * np.sum( np.log( (n / h) * (1. + np.log(h / h_low))*2. ) ))**0.5
    kappa = - np.log(4.**d * (2.*np.pi)**0.5)
    thresh = v + ((4.*d - 1)*np.log(v) + kappa + tau)/v
    return thresh

def adascan_mod_thresh(n,h,h_low,tau):
    d = len(h)
    v = (2. * np.sum( np.log( n / h )))**0.5
    kappa = - np.log(4.**d * (2.*np.pi)**0.5)
    thresh = v + ((4.*d - 1)*np.log(v) + kappa + tau)/v
    return thresh
        
def adascan_pvalue(n,h,h_low,s):
    d = len(h)
    v = (2. * np.sum( np.log( (n / h) * (1. + np.log(h / h_low))*2. ) ))**0.5
    kappa = - np.log(4.**d * (2.*np.pi)**0.5)
    tau = (s - v)*v - (kappa + (4.*d - 1.)*np.log(v))
    alpha = 1 - np.exp(-np.exp(-tau))
    return alpha

def adascan_ts(n,h,h_low,s):
    d = len(h)
    v = (2. * np.sum( np.log( (n / h) * (1. + np.log(h / h_low))*2. ) ))**0.5
    kappa = - np.log(4.**d * (2.*np.pi)**0.5)
    tau = (s - v)*v - (kappa + (4.*d - 1.)*np.log(v))
    alpha = np.exp(-tau)
    return alpha

def adascan_mod_ts(n,h,h_low,s):
    d = len(h)
    v = (2. * np.sum( np.log( n / h )))**0.5
    tau = (s - v)*v - (4.*d - 1)*np.log(v)
    alpha = np.exp(-tau)
    return alpha

def adascan_2d(y,h_low = 1.):
    n = y.shape[0]
    max_c = []
    L = int(n/(np.e*h_low))
    for r1 in np.arange(L) + h_low:
        for r2 in np.arange(L) + h_low:
            h = np.array([r1,r2])
            Z = np.max(conv_rect(y,2,h)) / (r1*r2)**0.5
            max_c.append(adascan_pvalue(n,h,h_low,Z))
    return 0, max_c

def adascan_ts_2d(y,h_low = 1.):
    n = y.shape[0]
    max_c = []
    L = int(n/(np.e*h_low))
    for r1 in np.arange(L) + h_low:
        for r2 in np.arange(L) + h_low:
            h = np.array([r1,r2])
            Z = np.max(conv_rect(y,2,h)) / (r1*r2)**0.5
            max_c.append(adascan_ts(n,h,h_low,Z))
    return 0, max_c

def adascan_mod_2d(y,h_low = 1.):
    n = y.shape[0]
    max_c = []
    L = int(n/(np.e*h_low))
    for r1 in np.arange(L) + h_low:
        for r2 in np.arange(L) + h_low:
            h = np.array([r1,r2])
            Z = np.max(conv_rect(y,2,h)) / (r1*r2)**0.5
            max_c.append(adascan_mod_ts(n,h,h_low,Z))
    return 0, max_c

def dyad_2d(y):
    n1, n2 = y.shape
    L1 = int(np.ceil(np.log2(n1)))
    L2 = int(np.ceil(np.log2(n2)))
    dyad = {}
    for i1 in range(L1):
        for i2 in range(L2):
            N1 = 2**(L1 - i1)
            N2 = 2**(L2 - i2)
            dyad[i1*L1 + i2] = np.zeros((N1,N2))
            if i1 == 0 and i2 == 0:
                dyad[i1*L1 + i2][0:n1,0:n2] = y
            elif i2 == 0:
                for t1 in range(N1):
                    for t2 in range(N2):
                        dyad[i1*L1 + i2][t1,t2] = dyad[(i1-1)*L1 + i2][2*t1,t2] + dyad[(i1-1)*L1 + i2][2*t1 + 1,t2]
            else:
                for t1 in range(N1):
                    for t2 in range(N2):
                        dyad[i1*L1 + i2][t1,t2] = dyad[i1*L1 + (i2 - 1)][t1,2*t2] + dyad[i1*L1 + (i2 - 1)][t1,2*t2 + 1]
    return dyad

def eps_adascan_2d(y, eps = .49, h_low = 1,verbose = False, dyad=False):
    if dyad == False:
        dyad = dyad_2d(y)
    d = 2
    n1, n2 = y.shape
    assert n1 == n2
    n = n1
    L1 = int(np.ceil(np.log2(n1)))
    alpha = eps**2. / (4.*d)
    a_low = max(int(np.log2(h_low * alpha)),0)
    a_high = max(int(np.log2(n / np.e * alpha)),1)
    max_c = []
    rhos = []
    for a1 in range(a_low,a_high):
        for a2 in range(a_low,a_high):
            if a1 == 0:
                H1 = np.arange(int(2./alpha + 1.)) + 1
            else:
                H1 = np.arange(int(1./alpha - 1e-6), int(2./alpha + 1.)) + 1
            if a2 == 0:
                H2 = np.arange(int(2./alpha + 1.)) + 1
            else:
                H2 = np.arange(int(1./alpha - 1e-6), int(2./alpha + 1.)) + 1
            for h1 in H1:
                for h2 in H2:
                    rho1 = h1 * 2**a1
                    rho2 = h2 * 2**a2
                    if rho1 >= h_low and rho2 >= h_low:
                        Zs = conv_rect(dyad[a1*L1 + a2],2,[h1,h2]) / (rho1 * rho2)**0.5
                        pval = adascan_pvalue(n,np.array([rho1,rho2]),h_low,np.max(Zs))
                        max_c.append(pval)
                        rhos.append([rho1,rho2])
    return rhos, max_c

def eps_scan_2d(y, eps = .49, h_low = 1,verbose = False, dyad=False):
    if dyad == False:
        dyad = dyad_2d(y)
    d = 2
    n1, n2 = y.shape
    assert n1 == n2
    n = n1
    L1 = int(np.ceil(np.log2(n1)))
    alpha = eps**2. / (4.*d)
    a_low = max(int(np.log2(h_low * alpha)),0)
    a_high = max(int(np.log2(n / np.e * alpha)),1)
    max_c = []
    rhos = []
    for a1 in range(a_low,a_high):
        for a2 in range(a_low,a_high):
            if a1 == 0:
                H1 = np.arange(int(2./alpha + 1.)) + 1
            else:
                H1 = np.arange(int(1./alpha - 1e-6), int(2./alpha + 1.)) + 1
            if a2 == 0:
                H2 = np.arange(int(2./alpha + 1.)) + 1
            else:
                H2 = np.arange(int(1./alpha - 1e-6), int(2./alpha + 1.)) + 1
            for h1 in H1:
                for h2 in H2:
                    rho1 = h1 * 2**a1
                    rho2 = h2 * 2**a2
                    if rho1 >= h_low and rho2 >= h_low:
                        Zs = conv_rect(dyad[a1*L1 + a2],2,[h1,h2]) / (rho1 * rho2)**0.5
                        pval = mult_scan_ts(n,d,h_low,np.max(Zs))
                        max_c.append(pval)
                        rhos.append([rho1,rho2])
    return rhos, max_c
































