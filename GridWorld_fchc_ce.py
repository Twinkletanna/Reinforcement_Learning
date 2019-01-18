#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 20:27:36 2018

@author: twinkle
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
import time
from tqdm import *
   
class GridWorld:
    def __init__(self):
        self.grid=[0,1,2,3,4,5,6,7,8,9,10,11,-1,12,13,14,15,-1,16,17,18,19,20,21,22]
        self.grid=np.reshape(self.grid,(5,5))
        self.reward=np.zeros((5,5))
        self.reward[4][2]=-10
        self.reward[4][4]=10
        self.mean=0
        self.K=20
        self.Ke=5
        self.N=10
        self.e=0.0005
        self.n=23*4
        self.ce_trials=10
        self.fchc_trials=10
        self.theta=np.zeros(self.n)
        self.cov=np.eye(self.n)*5
        self.sigma=1
        
        
    def attemp_action(self,state,selected_action):
    
        
        env=[0.8,0.05,0.05,0.1]
        env_name=['move','left','right','stay']
        #actions_names=['AU','AD','AR','AL']        
    
        change=np.random.choice(4,1,p=env)
        
        
        a=state[0]
        b=state[1]
        
        if selected_action==0 and change==0 :
            new_state=[a-1,b]
        elif selected_action==0 and change==1:
            new_state=[a,b-1]
        elif selected_action==0 and change==2:
            new_state=[a,b+1]
        elif selected_action==0 and change==3:
            new_state=[a,b]
            
        elif selected_action==1 and change==0:
            new_state=[a+1,b]
        elif selected_action==1 and change==1:
            new_state=[a,b+1]
        elif selected_action==1 and change==2:
            new_state=[a,b-1]            
        elif selected_action==1 and change==3:
            new_state=[a,b] 
    
        elif selected_action==2 and change==0:
            new_state=[a,b+1]
        elif selected_action==2 and change==1:
            new_state=[a-1,b]
        elif selected_action==2 and change==2:
            new_state=[a+1,b]
        elif selected_action==2 and change==3:
            new_state=[a,b]         
            
        elif selected_action==3 and change==0:
            new_state=[a,b-1]
        elif selected_action==3 and change==1:
            new_state=[a-1,b]        
        elif selected_action==3 and change==2:
            new_state=[a+1,b]        
        elif selected_action==3 and change==3:
            new_state=[a,b]        
            
                
        if new_state[0]<0 or new_state[0]>4:
            new_state=[a,b]
        if new_state[1]<0 or new_state[1]>4:
            new_state=[a,b]
        if new_state==[2,2] or new_state==[3,2]:
            new_state=[a,b]
    
        return new_state    
    
        
    def run_optimal(self,policy):
        state=[0,0]
        sum=0
        i=0
        gamma=0.9
        while state!=[4,4] and i<100:
            state_num=self.grid[state[0]][state[1]]
            selected_action=np.argmax(policy[state_num])
            new_state=self.attemp_action(state,selected_action)
            reward=self.reward[new_state[0],new_state[1]]
            state=new_state
            sum+=reward*gamma**i
            if reward not in (0,-10,10):
                print(reward)
            i+=1
        return sum 

    
    def evaluate(self,theta):
        rewards=[]
        for i in range(0,self.N):
            theta_mod=np.reshape(theta,(23,4))
            policy=(np.exp(theta_mod)) / (np.sum(np.exp(np.reshape(theta_mod,(23,4))),axis=1).reshape(23,-1))
            rewards.append(self.run_optimal(policy))
        return rewards

    def run_trials():    
        trials=0
        returns_avg_trials =[]
        self.ce_trials=500
        self.iters=50
        self.K=100
        self.Ke=10
        self.N=50
        self.e=0.001
        for trials in tqdm(range(self.ce_trials)):            
            reward,time=self.ce()
            print('trial: ',trials,'time: ',time,'mean: ',np.mean(reward))
            np.save('./numpyfiles/gw_ce_1_'+str(trials)+'.npy')

        
  
    def ce(self):
        start=time.time()
        returns=[]
        J=[]
        thetas=[]
        self.theta=np.random.rand(self.n)
        self.cov=np.eye(self.n)*150
        draws=np.random.multivariate_normal(self.theta,self.cov,(self.iters,self.K))
        for l in range(0,self.iters):
            for i in range(0,self.K):
                reward=self.evaluate(draws[l][i])
                returns.extend(reward)
                J.append(np.mean(reward))
                thetas.append(theta_new)
            arg=np.argsort(J)
            J_new=np.array(J[arg][-self.K:])
            thetas_np=np.array(thetas)
            selected_thetas=thetas_np[arg][-self.Ke:]
            self.theta=np.mean(selected_thetas,axis=0).reshape(-1)
            new_cov=self.cov*0
            for each in selected_thetas:
                new_cov+=(each-self.theta)*(each-self.theta).T
            self.cov=(self.e*np.eye(self.n)+ new_cov )/(self.e+self.Ke)
        returns_avg_trials.append(returns)
        end=time.time()
      return returns, end-start
    
    
    def fchc(self):
        # self.sigma=np.random.rand(1)
        # self.cov=np.eye(92)*self.sigma
        trials=0
        returns_avg_trials=[]

        # while trials<self.fchc_trials:
        for trials in tqdm(range(self.fchc_trials)):
            # trials+=1
            print(trials)
            start=time.time()
        
            self.theta=np.random.rand(92)
            rewards=self.evaluate(self.theta)
            itr=0
            j_mean=np.mean(rewards[-1])
            theta_mean=self.theta
            rewards=[]
            while itr<self.iters:
                itr+=1
                theta_new=np.random.multivariate_normal(self.theta,self.cov)
                reward=self.evaluate(theta_new)
                j=np.mean(reward)
                rewards.extend(reward)
                if j>j_mean:
                    j_mean=j
                    theta_mean=theta_new
                    self.theta=theta_mean
 
                if itr%50==0:
                    print(itr,np.round(np.mean(reward),4),np.round(j_mean))
 
            returns_avg_trials.append(rewards)
            end=time.time()
            if trials%10==0 or trials==1:
                np.save('./numpyfiles/gw_fchc_local_1_highsigma_1_'+str(trials)+'.npy',returns_avg_trials)
        
            print('trial: ',trials,' time: ',end-start)

        r=np.mean(returns_avg_trials,axis=0)
        std=np.std(returns_avg_trials,axis=0)
        return returns_avg_trials,r,std