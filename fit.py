import numpy as np
import scipy.optimize as opt
import sys

# model:
# a series of observations, x, are made iid from the model
#  x_i = a + b_i
# where a and b_i are both normally distributed
# a has unknown mean (amu) and variance (avar) which are the model parameters
# b_i has 0 mean and known variance (bvar), which varies per observation

# load data from text file:
# first row (whitespace separated) are recorded values
# second row (whitespace separated) are the corresponding estimated 
#  observational noise variances
D = np.loadtxt(sys.argv[1])

# remove infinite variance values
D = D[:,np.isfinite(D[1,:])]
x = D[0,:]
bvar = D[1,:]
m = D.shape[1]

# multiplicative factor around the naive estimate to define
# the region in which to search for MLE estimate
spread = 100

# fits the mean of the model, given the variance of the model
# (can be done in closed form)
def fitmu(x,bvar,avar,dens=None):
    if dens is None:
        dens = bvar + avar
    return np.sum(x/dens)/np.sum(1.0/dens)

# returns negative llh of the data
# ignores the sqrt(2pi) part (should add (#x)*log(sqrt(2pi)) to
#  get true negative log-likelihood)
def negllh(x,bvar,avar,mu=None):
    dens = bvar + avar
    if mu is None:
        mu = fitmu(D,avar,dens)
    return np.sum(np.log(dens) + np.square(x - mu)/dens)/2

# rough estimates of the variance of a (avar)
# to be used to find region in which to search for MLE
naivemu = np.sum(D[0,:]/D[1,:])/np.sum(1.0/D[1,:])
naivevar = np.sum(np.square(D[0,:]-naivemu)/D[1,:])/np.sum(1.0/D[1,:])

# REAL WORK NEXT LINE:
# line search for avar (with amu calculated analytically from avar and data)
# for minimum of negllh
optsol = opt.minimize_scalar(lambda avar : negllh(x,bvar,avar), bounds=(naivevar/spread,naivevar*spread),method='bounded')

if not optsol.success:
    print("optimization failed")
mleavar = optsol.x
mleamu = fitmu(x,bvar,mleavar)
print(mleamu,mleavar,negllh(D,mleavar,mleamu))
