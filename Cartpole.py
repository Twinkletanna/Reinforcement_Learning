from RLHW2 import *

cp=CartPole()
cp.ce_trials=500
cp.iters=20
cp.N=1
cp.n=4
cp.K=50
cp.Ke=5
cp.e=0.01
# cp.theta
r,mean_r,std=cp.ce()
np.save('./numpyfiles/cartpole_ce.npy',r)