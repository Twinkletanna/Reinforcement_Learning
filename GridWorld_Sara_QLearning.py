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
import random
from itertools import product as comb
   
class GridWorld:
    def __init__(self):
        self.grid=[0,1,2,3,4,5,6,7,8,9,10,11,-1,12,13,14,15,-1,16,17,18,19,20,21,22]
        self.grid=np.reshape(self.grid,(5,5))
        self.reward=np.zeros((5,5))
        self.reward[4][2]=-10
        self.reward[4][4]=10
        self.q=np.random.rand(23,4)+10
        self.gamma=1     
        self.eps=0.001

        
    def attemp_action(self,state,selected_action):
        env=[0.8,0.05,0.05,0.1]
        env_name=['move','left','right','stay']
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
            
        # print(selected_action,change)
                
        if new_state[0]<0 or new_state[0]>4:
            new_state=[a,b]
        if new_state[1]<0 or new_state[1]>4:
            new_state=[a,b]
        if new_state==[2,2] or new_state==[3,2]:
            new_state=[a,b]
    
        return new_state    

    def egreedy(self,state_num):
        check=np.random.rand()
        if check<self.eps:
            a=np.random.choice(4,1)
        else:
            # print(self.q[state_num].shape)
            a=np.argmax(self.q[state_num])
            # print(a)
        # print(a)
        return a
    
    def run_sarsa(self,state,alpha,eps):
        state=[0,0]
        self.eps=eps
        history=[state]
        sum=0
        i=0
        gamma=0.9

        selected_action=self.egreedy(state)  

        while state!=[4,4] and i<1000:

            new_state=self.attemp_action(state,selected_action)
            state_num=self.grid[state[0]][state[1]]
            new_state_num=self.grid[new_state[0]][new_state[1]]
            reward=self.reward[new_state[0],new_state[1]]
            next_action=self.egreedy(state_num)  

            history.append(state)
            sum+=reward*gamma**i
            if reward not in (0,-10,10):
                print(reward)
            i+=1

            if i%25==0:
                alpha-=alpha/i
            #     alpha=alpha/i

            delta=reward+self.gamma*self.q[new_state_num][next_action]-self.q[state_num][selected_action]
            self.q[state_num][selected_action]+=alpha*delta

            state=new_state
            selected_action=next_action
            self.alpha=alpha

        return sum , history

    
    def sarsa_trial(self,alpha,eps,n):
        r=[]
        self.alpha=alpha
        self.q=np.random.rand(23,4)+10
        for i in range(0,n):
            returns,history=self.run_sarsa([0,0],alpha,eps)
            r.append(returns)
        return r


    def avgtrials(self,alpha,eps,t,n):
        r=[]
        print('Gridworld')
        for i in range(t):
            print('trial',i)
            x=self.sarsa_trial(alpha,eps,n)
            r.append(x)
            plt.plot(x)
            plt.savefig('./gwgraph/sgw'+str(i)+str(alpha)+str(eps)+'.jpg')
            plt.clf()
            np.save('./gwgraph/sgw_trial'+str(i)+str(alpha)+str(eps)+'.npy',x)

        print('done')
        r=np.asarray(r)
        print(r.shape)
        vals=np.mean(r,axis=0)
        print(vals.shape)
        plt.plot(vals)
        plt.savefig('./GridWorld'+str(alpha)+str(eps)+'.jpg')
        plt.clf()
        # plt.show()
                
    def qlearn_ep(self,state,alpha,eps):
        state=[0,0]
        self.eps=eps
        self.alpha=alpha
        history=[state]
        sum=0
        i=0
        gamma=0.9

        while state!=[4,4] and i<1000:

            state_num=self.grid[state[0]][state[1]]
            selected_action=self.egreedy(state_num)
            # print(selected_action)
            new_state=self.attemp_action(state,selected_action)
            new_state_num=self.grid[new_state[0]][new_state[1]]
            reward=self.reward[new_state[0],new_state[1]]
    
            # state=new_state
            # next_action=self.egreedy(state_num)  

            history.append(state)
            sum+=reward*gamma**i
            i+=1

            if i %25==0 and i>0:
                alpha-=alpha/i

            delta=reward+self.gamma*np.max(self.q[new_state_num])-self.q[state_num][selected_action]
            self.q[state_num][selected_action]+=alpha*delta
            self.alpha=alpha
            state=new_state

        # print(state)
        return sum , history


    def run_qlearn(self,alpha,eps,t,n):
        all_trials=[]

        for j in range(t):
            print('trial',str(j))
            one_trial=[]
            self.q=np.random.rand(23,4)+10
            self.alpha=alpha
            self.eps=eps

            for i in range(n):
                r,hist=self.qlearn_ep([0,0],self.alpha,eps)
                print(r,self.alpha)
                one_trial.append(r)


            plt.plot(one_trial)
            plt.savefig('./gwgraph/qgw_trial'+str(j)+str(alpha)+str(eps)+'.jpg')
            plt.clf()
            np.save('./gwgraph/qgw_trial'+str(j)+str(alpha)+str(eps)+'.npy',one_trial)

            all_trials.append(one_trial)

        avgd_trials=np.mean(all_trials,axis=0)
        plt.plot(avgd_trials)
        plt.savefig('./GridWorld_Qlearn'+str(alpha)+str(eps)+'.jpg')
        plt.clf()


# 
# g=GridWorld()
# g.avgtrials()
# g.run_qlearn()

# c=CartPole()
# x=c.run_sarsa(1e-5,.1)
# c.run_episode_sarsa(1e-5,1)
# x=c.run_episode_q(0.5,1)
# x=c.run_qlearn(1e-5,.1)
# print(x)