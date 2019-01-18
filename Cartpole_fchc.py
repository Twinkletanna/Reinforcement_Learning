from RLHW2 import *

cp=CartPole()
cp.fchc_trials=500
cp.iters=500
cp.N=1
cp.n=4
cp.sigma=0.8
cp.cov=np.eye(cp.n)*cp.sigma
r,mean_r,std=cp.fchc()
name="./numpyfiles/cp_fchc.npy"
print('saved ',name)
np.save(name)