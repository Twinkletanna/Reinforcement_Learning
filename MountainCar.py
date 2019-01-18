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
   
class MountainCar():
    """docstring for ClassName"""
    def __init__(self):
        self.state=()        
        self.time=0 
        self.state=(-0.5,0)
        self.w=np.zeros((16,2))
        self.q=np.array((16,2))
        self.k=5
        self.gamma=1
        self.N=10    
        self.eps=1

    def take_action(self,state,selected_action):
        x=state[0]
        v=state[1]

        next_v=v+0.001*selected_action-0.0025*math.cos(3*x)
        next_x=x+next_v

        if next_x<-1.2 :
            next_x=-1.2
            next_v=0
        if next_x>0.5:
            next_x=0.5
            next_v=0


        return [next_x,next_v]
    

    def get_phi(self,state,k,z):
        c=list(comb(np.arange(k),repeat=2))
        c=np.asarray(c)
        s=self.normalise(state)
        phi=np.cos( math.pi * np.dot(c,s) ).ravel()
        phi_left= np.array([phi,z,z])
        phi_n=np.array([z,phi,z])
        phi_right=np.array([z,z,phi])

        return phi_left,phi_n,phi_right

    # def stop_cond(self,state):
    #     if self.state[2]>=math.pi/2 or self.state[2]<=-math.pi/2:
    #         return True
    #     if self.state[0]<=-3 or self.state[0]>=3:
    #         return True
    #     return False

    def egreedy(self,phi_left,phi_n,phi_right,w):
        check=np.random.rand()
        if check<self.eps:
            a=np.random.choice([-1,0,1],1)
            # print('random')
        else:
            # print('greedy')
            q_left=(self.w*phi_left).sum()
            q_n=(self.w*phi_n).sum()
            q_right=(self.w*phi_right).sum()

            a=np.argmax([q_left,q_n,q_right])-1

        if a==-1:
            phi=phi_left
        elif a==0:
            phi=phi_n
        elif a==1:
            phi=phi_right

        return a, phi

    def run_episode_sarsa(self,alpha,e,n):
        self.alpha=alpha
        self.eps=e
        k=6
        #change k
        # print('Fourier order:',k-1)
        
        self.w=np.zeros((3,k**2))
        z=np.zeros(k**2)
        returns=[]
        c=0
        for j in range(0,n):#200 eps
            self.state=(-0.5,0)
            self.time=0
            hist=[]
            reward=0
            i=0

            # if j%10==0:
                # self.eps/=2

            phi_left,phi_n,phi_right=self.get_phi(self.state,k,z)
            selected_action,this_phi=self.egreedy(phi_left,phi_n,phi_right,self.w) 

            # selected_action=np.random.choice([0,1],1)
            # print(selected_action)

            while self.state[0]!=0.5  and i<50000: 
                c+=1
                i+=1
                state=self.state
                print(state,i)
                self.state=self.take_action(self.state,selected_action)

                reward-=1

                phi_left,phi_n,phi_right=self.get_phi(state,k,z)
                next_phi_left,next_phi_n,next_phi_right=self.get_phi(self.state,k,z)

                if selected_action==-1:
                    this_phi=phi_left
                elif selected_action==0:
                    this_phi=phi_n
                else:
                    this_phi=phi_right

                next_action, next_phi=self.egreedy(next_phi_left,next_phi_n,next_phi_right,self.w)   


                #sarsa UPDATE
                # print(this_phi.shape,self.w.shape)
                self.q=(self.w*this_phi).sum()
                next_q=(self.w*next_phi).sum()
                delta=1+self.gamma*next_q-self.q
                self.w+=self.alpha*this_phi*delta

                #END OF sarsa UPDATE

                selected_action=next_action


            # print(reward)
            returns.append(reward)
            # if reward<=returns[-1] and c>1:
                # self.eps=self.eps*10
                # print('inc eps')

        return returns
    

    def run_episode_q(self,alpha,e,n):
        self.alpha=alpha
        self.eps=e
        k=6
        #change k
        # print('Fourier order:',k-1)
        self.w=np.zeros((3,k**2))
        z=np.zeros(k**2)
        returns=[]
        c=0
        for j in range(0,n):#200 eps
            self.state=(-0.5,0)
            self.time=0
            hist=[]
            reward=0
            i=0

            # if j%10==0:
                # self.eps/=2


            while self.state[0]!=0.5  and i<50000: 
                c+=1
                self.time+=0.02
                i+=1



                phi_left,phi_n,phi_right=self.get_phi(self.state,k,z)
                selected_action,this_phi=self.egreedy(phi_left,phi_n,phi_right,self.w) 

                state=self.state
                self.state=self.take_action(self.state,selected_action)


                # phi_left,phi_right=self.get_phi(state,k,z)
                next_phi_left,next_phi_n,next_phi_right=self.get_phi(self.state,k,z)

                reward-=1

                if selected_action==-1:
                    this_phi=phi_left
                elif selected_action==0:
                    this_phi=phi_n
                else:
                    this_phi=phi_right

                # next_action, next_phi=self.egreedy(phi_left,phi_right,self.w)   

                #q UPDATE
                self.q=(self.w*this_phi).sum()

                next_q_left=(self.w*next_phi_left).sum()                
                next_q_n=(self.w*next_phi_n).sum()
                next_q_right=(self.w*next_phi_right).sum()

                # print(self.q.shape,next_phi_left.shape,next_q_left.shape)
                # if next_q_left>next_q_right:
                #     next_q=next_q_left

                # else:
                #     next_q=next_q_right

                # print(next_q_left,next_q_n,next_q_right
)                next_q=np.max([next_q_left,next_q_n,next_q_right])




                delta=1+self.gamma*next_q-self.q
                # print(self.q.shape,next_q.shape,delta.shape)
                # print('update:',np.sum(delta))
                self.w+=self.alpha*this_phi*delta
                # print('done update')           
#                 if c %100==0:
#                     if self.eps>1e-4:
#                         self.eps-=self.eps/2
#                         # self.eps/=10
#                         # print('changed eps')
# # 
#                 if c%500==0:
#                     # alpha-=alpha/i
#                     if self.alpha>1e-8:
#                         self.alpha-=self.alpha/50
#                         # print('changed alpha')


                #END OF q UPDATE

                # selected_action=next_action

            # print(reward,self.alpha)
            returns.append(reward)
            # if reward<=returns[-1] and c>1:
                # self.eps=self.eps*10
                # print('inc eps')

        return returns




    def get_poly_phi(self,state,k,z):
        c=list(comb(np.arange(k),repeat=4))
        c=np.asarray(c)
        s=self.normalise(state)
        # phi=np.cos( math.pi * np.dot(c,s) ).ravel()

        poly_vec=[]
        for each in c:
            poly_part=1
            for i in range(each):
                poly_part*=s**each[i]
            poly_vec.append(poly_part)
        poly_vec=np.array(poly_vec)

        phi_left= np.array([poly_vec,z])
        phi_right=np.array([z,poly_vec])

        return phi_left,phi_right


    



    def normalise(self,history):
        a=(history[0]+1.2)*1.0/1.7;
        b=(history[1]++0.07)/0.14;
        normalised_state=np.array([a,b])
        return normalised_state

   
    def run_sarsa(self,alpha,eps,t,n):
        r=[]
        for i in range(t):
            print('Sarsa trial',i)
            x=self.run_episode_sarsa(alpha,eps,n)
            print(x)
            r.append(x)
            plt.plot(x)
            plt.savefig('./mc/cpsarsa'+str(i)+str(alpha)+str(eps)+'.jpg')
            plt.clf()
            np.save('./mc/cpsarsa_trial'+str(i)+'.npy',x)

        print('done')
        r=np.asarray(r)
        print(r.shape)
        vals=np.mean(r,axis=0)
        print(vals.shape)
        plt.plot(vals)
        plt.savefig('./mcSarsa'+str(alpha)+str(eps)+'.jpg')
        plt.clf()

        # plt.show()
        return vals

    def run_qlearn(self,alpha,eps,t,n):
        r=[]
        for i in range(t):
            print('Q trial',i)
            x=self.run_episode_q(alpha,eps,n)
            r.append(x)
            plt.plot(x)
            plt.savefig('./mc/cpq'+str(i)+str(alpha)+str(eps)+'.jpg')
            plt.clf()
            np.save('./cpq/cpqlearn_trial'+str(i)+'.npy',x)


        # print('done')
        r=np.asarray(r)
        # print(r.shape)
        vals=np.mean(r,axis=0)
        print(vals.shape)
        plt.plot(vals)
        plt.savefig('./mcQ'+str(alpha)+str(eps)+'.jpg')
        plt.clf()
        # plt.show()
        return vals

c=MountainCar()
# c.run_sarsa(0.5,1,1,10)
c.run_qlearn(0.5,1,1,10)
