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
        self.v=np.zeros(23)
        self.alpha=0.1
        self.gamma=1     
        
        
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
            
                
        if new_state[0]<0 or new_state[0]>4:
            new_state=[a,b]
        if new_state[1]<0 or new_state[1]>4:
            new_state=[a,b]
        if new_state==[2,2] or new_state==[3,2]:
            new_state=[a,b]
    
        return new_state    
    
    def pretty_print(self,lst):
        for i in range(len(lst)):
            print(lst[i])
        print()

    def run_o(self,alpha):
        values = [] #5x5
        td_errors = []
        for i in range(5):
            temp = [0.0]*5
            values.append(temp)
        #values[0][1]=1
        #values[1][0]=1
        num_eps = 100+100
        self.gamma = 0.9
        for ep in range(num_eps):
            # This lifts code from simulate(). Maybe club them?
            state = [0,0]
            time_step = 0
            while(state!=[4,4]):
                #action=np.random.choice(4,1) #In HW3 we always choose random action
                action=random.randrange(0,4)
                new_state = self.attemp_action(state,action) #s_fi
                
                r_t = self.reward[new_state[0]][new_state[1]]
                td_error = r_t + self.gamma*values[new_state[0]][new_state[1]] - values[state[0]][state[1]]
                if ep<100:
                    values[state[0]][state[1]]+= alpha*td_error
                else:
                    td_errors.append(td_error**2)
                #if time_step<50:
                #    print(s_in[0],s_in[1],action,r_t,s_fi[0],s_fi[1],td_error)
                time_step+=1
                state=list(new_state)
            #print("*******")
            # self.pretty_print(values) #Uncomment if you want to print values for the gridworld.
        return np.mean(td_errors)


    def run_optimal(self,alpha):
        # self.alpha=alpha
        td_errors=[]
        v=np.zeros(23)
        for j in range(0,200):        

            state=[0,0]
            history=[state]
            sum=0
            i=0
            gamma=0.9
            rewards=[]
            
            
            while state!=[4,4] :
                state_num=self.grid[state[0]][state[1]] #number b/w [0,22]
                # selected_action=np.argmax(policy[state_num])
                selected_action=np.random.choice(4,1) #random action          
                new_state=self.attemp_action(state,selected_action) #S_t+1
                reward=self.reward[new_state[0]][new_state[1]] #R_t
                # history.append(state)
                # rt=reward*(gamma**i)
                # sum+=rt
                # rewards.append(reward*gamma**i)
                # if reward not in (0,-10,10):
                    # print(reward)
                # i+=1

                #TD UPDATES
                current=self.grid[state[0]][state[1]]
                next_state=self.grid[new_state[0]][new_state[1]]


                x=(reward+self.gamma*v[next_state] -v[current])
                if j<100 :
                    v[current]+=x*alpha
                else:
                    td_errors.append(x**2)

                state=new_state

        td_errors=np.asarray(td_errors)
        return np.mean(td_errors)

                

    def plot(self):
        vals=[]
        # x=[.0000001,.000001,.00001,.0001,.001,.01,.1,1]
        # x=[.000001,.00001,.0001,.001,.01,.1]
        x = [1e-6,1e-5,0.0001,0.001,0.01,0.1,1,10,100]
        for i in x:
            vals.append(self.run_optimal(i))
            # vals.append(self.run_o(i))
        print(np.log10(x))
        print(vals)
        #plt.plot(np.log10(x), vals)
        plt.semilogx(x,vals)
        # plt.show()
        plt.title('Gridworld TD')
        plt.xlabel('Step size (log)')
        plt.ylabel('TD Errors')
        plt.savefig('./GridWorld.jpg')
        plt.clf()
        return vals


class CartPole():
    """docstring for ClassName"""
    def __init__(self):
        self.state=()        
        self.time=0 
        self.state=(0,0,0,0)
        self.w=np.zeros((16,4))
        self.v=np.array((16,4))
        self.k=3
        self.gamma=1
        self.N=10    

    def take_action(self,state,selected_action):
        g=9.8
        m_cart=1
        m_pole=0.1
        l=0.5
        if selected_action==0:
            selected_action=-1
        F=10*selected_action
        t=0.02

        new_x=state[0]+state[1]*t
        new_theta=state[2]+state[3]*t

        angle=state[2]

        num=(g*math.sin(angle)+math.cos(angle)*( (-F-m_pole*l*state[3]**2*math.sin(angle)) / (m_cart+m_pole) ))
        den=l*( 4.0/3- (m_pole*(math.cos(angle))**2 )/(m_pole+m_cart) )
        ang_acc=num/den
        acc=(F+m_pole*l*(state[3]**2*math.sin(angle)  - ang_acc*math.cos(angle)  ))/(m_cart+m_pole) 

        new_angv=state[3]+ang_acc*t
        new_v=state[1]+acc*t
        new_state=(new_x,new_v,new_theta,new_angv)
#         print(new_x,new_v,new_theta,new_angv,ang_acc,acc)

        return new_state
    

    def run_episode(self,k,alpha):
        self.alpha=alpha
        self.k=k
        #change k
        print('Fourier order:',k)
        td_errors=[]
        self.w=np.zeros(k**4)
        for j in range(0,200):#200 eps
            self.state=(0,0,0,0)
            self.time=0

            hist=[]
            reward=0
            i=0
            while self.time<20.2  : 
                self.time+=0.02
                selected_action=np.random.choice([0,1],1)   
                i+=1
                state=self.state
                self.state=self.take_action(self.state,selected_action)
                if self.state[2]>=math.pi/2 or self.state[2]<=-math.pi/2:
                    break
                if self.state[0]<=-3 or self.state[0]>=3:
                    break


                #TDUPDATE
                c=list(comb(np.arange(k),repeat=4))
                c=np.asarray(c)

                # state=history[i]
                s=self.normalise(state)
                # print(c.shape,s.shape)
                # x=np.dot(c,s)
                # print(x.shape)
                phi=np.cos( math.pi * np.dot(c,s) ).ravel() 

                next_state=self.state
                next_s=self.normalise(next_state)
                next_phi=np.cos( math.pi * np.dot(c,next_s)).ravel()

                #print(phi.shape)
                # print(next_phi.shape)             
                self.v=np.dot(self.w.T , phi) 
                next_v=np.dot(self.w.T,next_phi)
                delta=1+self.gamma*next_v-self.v

                # x=np.dot(phi,delta)
                if j<100:
                    # print(self.w.shape,phi.shape)
                    self.w+=self.alpha*phi*delta
                    # print(delta)
                else:
                    td_errors.append(delta) 

                #END OF TD UPDATE

                hist.append(np.array(self.state))
                reward+=1

        td_errors=np.array(td_errors)
        return np.mean(td_errors**2)
    

    def normalise(self,history):
        a=(history[0]+3)*1.0/6;
        b=(history[1]+10)/20;
        c=(history[2]+math.pi/2)/(math.pi)
        d=(history[3]+math.pi/2)/(math.pi)
        normalised_state=np.array([a,b,c,d])
        return normalised_state

   
    def get_all_tde(self):
        k=[4,6]
        # k = [4]
        alpha=[.0000001,.000001,.00001,.0001,.001,.01,.1]
        # alpha = [5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 0.00009, 0.0001,0.002,0.0025, 0.003]
        errors_for_k=[]
        for each in k:
            errors_for_alphas=[]
            for every in alpha:
                tde=self.run_episode(each,every)
                # print(tde)
                # print(type(tde))
                errors_for_alphas.append(tde)
            errors_for_k.append(errors_for_alphas)
        print (errors_for_k)

        plt.title('CartPole TD with Third order Fourier Basis')
        plt.xlabel('Step size (log)')
        plt.ylabel('TD Errors')
        plt.semilogx(alpha,errors_for_k[0])
        plt.savefig('./cartpole3.jpg')
        plt.clf()
        # '''
        plt.xlabel('Step size (log)')
        plt.ylabel('TD Errors')
        plt.title('CartPole TD with Fifth order Fourier Basis')
        plt.semilogx(alpha,errors_for_k[1])
        plt.savefig('./cartpole5.jpg')
        plt.clf()
        # '''
        return errors_for_k


g=GridWorld()
vals=g.plot()

# c=CartPole()
# vals2=c.get_all_tde()

# alpha=[.0000001,.000001,.00001,.0001,.001,.01,.1]
# np.save('./values.npy',[vals,vals2,alpha])

# plt.title('Gridwold and Cartpole TD Errors vs Alpha')
# plt.xlabel('Step size')
# plt.ylabel('TD Errors')
# plt.plot(np.log10(alpha),np.log10(vals))
# plt.plot(np.log10(alpha),np.log10(vals2[0]))
# plt.plot(np.log10(alpha),np.log10(vals2[1]))
# plt.legend(['Gridworld','Cartpole FourierBasis 3','Cartpole FourierBasis 5'],loc="upper left")
# plt.savefig('./all.jpg')
# plt.clf()