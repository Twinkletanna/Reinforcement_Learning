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

        part=0.001*selected_action-0.0025*math.cos(3*x)
        next_v=v+part
        next_x=x+next_v

        if next_x<-1.2 :
            next_x=-1.2
            next_v=0
        if next_x>0.5:
            next_x=0.5
            next_v=0

        if next_v<-0.07:
            next_v=-0.07
        if next_v>0.07:
            next_v=0.07

        # print(state,part,selected_action,next_x,next_v)

        return [next_x,next_v]

    def one_ep(self):
        #One episode with random policy
        self.state=(-0.5,0)        
        hist=[]
        reward=0
        i=0
        while self.state[0]<0.5  and i<50000: 
            i+=1
            # state=self.state
            selected_action=np.random.choice([-1,0,1],1)[0]
            # selected_action=-1
            self.state=self.take_action(self.state,selected_action)
            print(i,self.state,selected_action)
            # x=input()
            reward-=1

    def get_phi(self,state,k,z,flag=0):
        c=list(comb(np.arange(k),repeat=2))
        c=np.asarray(c)
        s=self.normalise(state)
        # print(c)
        phi=np.cos( math.pi * np.dot(c,s) ).ravel()
        if flag==1:
            return phi
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
            a=np.random.choice([-1,0,1],1)[0]
            # print('random')
        else:
            # print('greedy')
            q_left=(self.w*phi_left).sum()
            q_n=(self.w*phi_n).sum()
            q_right=(self.w*phi_right).sum()

            a=np.argmax([q_left,q_n,q_right])-1

        # a=1

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
        
        # print(n,)
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
            phi_left,phi_n,phi_right=self.get_phi(self.state,k,z)
            selected_action,this_phi=self.egreedy(phi_left,phi_n,phi_right,self.w) 

            while self.state[0]<0.5  and i<50000: 
                c+=1
                i+=1
                state=self.state
                self.state=self.take_action(self.state,selected_action)
                if i%1000==0:
                    print(i,state,selected_action)

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
                self.q=(self.w*this_phi).sum()
                next_q=(self.w*next_phi).sum()
                delta=-1+self.gamma*next_q-self.q
                self.w+=self.alpha*this_phi*delta

                #END OF sarsa UPDATE

                if c%100==0:
                    self.alpha-=self.alpha/c
                if c%50==0:
                    self.eps-=self.eps/c

                selected_action=next_action

            returns.append(reward)

        return returns
    
    def run_episode_sarsa_lambda(self,alpha,e,n,l=0.2):
        self.alpha=alpha
        self.eps=e
        k=6
        self.w=np.zeros((3,k**2))
        z=np.zeros(k**2)
        returns=[]
        c=0
        for j in range(0,n):#200 eps
            self.state=(-0.5,0)
            self.time=0
            self.eligibility=np.zeros(k**2)
            hist=[]
            reward=0
            i=0

            phi_left,phi_n,phi_right=self.get_phi(self.state,k,z)
            selected_action,this_phi=self.egreedy(phi_left,phi_n,phi_right,self.w) 
            while self.state[0]<0.5  and i<50000: 
                c+=1
                i+=1
                state=self.state
                self.state=self.take_action(self.state,selected_action)
                if i%1000==0:
                    print(i,state,selected_action)
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


                self.eligibility*=self.gamma*l
                self.eligibility+=phi_left[0]

                #sarsa UPDATE
                self.q=(self.w*this_phi).sum()
                next_q=(self.w*next_phi).sum()
                delta=-1+self.gamma*next_q-self.q
                self.w[selected_action+1,:]+=self.alpha*self.eligibility*delta
                #END OF sarsa UPDATE

                if c%100==0:
                    self.alpha-=self.alpha/c
                if c%50==0:
                    self.eps-=self.eps/c

                selected_action=next_action

            returns.append(reward)
        return returns






    def run_episode_q(self,alpha,e,n):
        self.alpha=alpha
        self.eps=e
        k=8
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

            while self.state[0]<0.5  and i<50000: 
                c+=1
                i+=1
                
                phi_left,phi_n,phi_right=self.get_phi(self.state,k,z)
                selected_action,this_phi=self.egreedy(phi_left,phi_n,phi_right,self.w) 
                state=self.state
                if i%1000==0:
                    print(i,self.state,selected_action)
            
                self.state=self.take_action(self.state,selected_action)
                next_phi_left,next_phi_n,next_phi_right=self.get_phi(self.state,k,z)
                reward-=1

                if selected_action==-1:
                    this_phi=phi_left
                elif selected_action==0:
                    this_phi=phi_n
                else:
                    this_phi=phi_right



                #q UPDATE
                self.q=(self.w*this_phi).sum()
                next_q_left=(self.w*next_phi_left).sum()                
                next_q_n=(self.w*next_phi_n).sum()
                next_q_right=(self.w*next_phi_right).sum()

                next_q=np.max([next_q_left,next_q_n,next_q_right])

                delta=-1+self.gamma*next_q-self.q
                self.w+=self.alpha*this_phi*delta
                #END OF q UPDATE

                if c%100==0:
                    self.alpha-=self.alpha/c
                if c%50==0:
                    self.eps-=self.eps/c

            returns.append(reward)
        return returns



    def run_episode_q_lambda(self,alpha,e,n,l=0.2):
        self.alpha=alpha
        self.eps=e
        k=8
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
            self.eligibility=np.zeros(k**2)


            while self.state[0]<0.5  and i<50000: 
                c+=1
                i+=1
                
                phi_left,phi_n,phi_right=self.get_phi(self.state,k,z)
                selected_action,this_phi=self.egreedy(phi_left,phi_n,phi_right,self.w) 
                state=self.state
                if i%1000==0:
                    print(i,self.state,selected_action)
            
                self.state=self.take_action(self.state,selected_action)
                next_phi_left,next_phi_n,next_phi_right=self.get_phi(self.state,k,z)
                reward-=1

                if selected_action==-1:
                    this_phi=phi_left
                elif selected_action==0:
                    this_phi=phi_n
                else:
                    this_phi=phi_right

                self.eligibility*=self.gamma*l
                self.eligibility+=phi_left[0]

                #q UPDATE
                self.q=(self.w*this_phi).sum()
                next_q_left=(self.w*next_phi_left).sum()                
                next_q_n=(self.w*next_phi_n).sum()
                next_q_right=(self.w*next_phi_right).sum()
                next_q=np.max([next_q_left,next_q_n,next_q_right])

                delta=-1+self.gamma*next_q-self.q
                self.w[selected_action+1,:]+=self.alpha*self.eligibility*delta
                #END OF q UPDATE

                if c%100==0:
                    self.alpha-=self.alpha/c
                if c%50==0:
                    self.eps-=self.eps/c


            returns.append(reward)
        return returns

    def softmax(self,phi,theta):
        action_vector=np.dot(theta,phi)
        # print(theta.shape,phi.shape,action_vector.shape)

        m=np.max(action_vector)
        new_vector=math.e**(action_vector-m)
        soft_vector=new_vector/np.sum(new_vector)
        a=int(np.random.choice(3,1,p=soft_vector))


        return a-1,soft_vector[a]



    def run_episode_ac(self,alpha,beta,l,n):
        self.alpha=alpha
        self.beta=beta
        k=4
        self.w=np.zeros(k**2)
        self.theta=np.zeros((3,k**2))
        z=np.zeros(k**2)
        returns=[]
        c=0
        for j in range(0,n):#200 eps
            self.state=(-0.5,0)
            self.time=0
            hist=[]
            reward=0
            i=0
            self.Veligibility=np.zeros(k**2)
            self.Teligibility=np.zeros((3,k**2 ) )


            while self.state[0]<0.5  and i<5000: 
                c+=1
                i+=1
                
                #Actor
                phi=self.get_phi(self.state,k,z,1)
                selected_action, do=self.softmax(phi,self.theta) 
                state=self.state
                # if i%100==0:
                    # print(i,self.state,selected_action)
                # x=input()
                self.state=self.take_action(self.state,selected_action)
                

                next_phi=self.get_phi(self.state,k,z,1)
                reward-=1

                #End Actor

                #Critic Update

                self.Veligibility*=self.gamma*l
                self.Veligibility+=phi

                self.v=np.dot(self.w.T , phi)
                next_v=np.dot(self.w.T,next_phi)
                # print(self.v)
                delta=-1+self.gamma*next_v-self.v

                self.w+=self.alpha*delta*self.Veligibility
                #End critic update

                #Actor UPDATE
                self.Teligibility*=self.gamma*l
                self.Teligibility[selected_action,:]+=1-do
                self.theta+=self.beta*delta*self.Teligibility
                #End actor update
               

                if c%100==0:
                    self.alpha-=self.alpha/c
                if c%50==0:
                    self.eps-=self.eps/c


            returns.append(reward)
            print(reward)
        return returns


    def run_reinforce(self,alpha,beta,l,n):
        self.alpha=alpha
        self.beta=beta
        k=4
        self.w=np.zeros(k**2)
        self.theta=np.zeros((3,k**2))
        z=np.zeros(k**2)
        returns=[]
        c=0
        for j in range(0,n):#200 eps
            self.state=(-0.5,0)
            self.time=0
            self.grad=np.zeros((3,k**2))
            hist=[]
            reward=0
            i=0
            # self.Veligibility=np.zeros(k**2)
            self.eligibility=np.zeros(k**2 )


            while self.state[0]<0.5  and i<5000: 
                c+=1
                i+=1
                
                phi=self.get_phi(self.state,k,z,1)
                selected_action, do=self.softmax(phi,self.theta) 
                state=self.state
                # if i%100==0:
                    # print(i,self.state,selected_action)
                # x=input()
                self.state=self.take_action(self.state,selected_action)
                next_phi=self.get_phi(self.state,k,z,1)
                reward-=1
                hist.append([phi,selected_action,next_phi,do,reward])


            for i, each in enumerate(hist):

                phi,selected_action,next_phi,do,r=each

                gt=reward-r+-1

                self.v=self.w.T*phi
                self.grad+=(gt-self.v)*-1*do
                self.grad[selected_action:,]+=(gt-self.v)

                self.eligibility*=self.gamma*l
                self.eligibility+=phi

                next_v=self.w.T*phi
                delta=r+self.gamma*next_v-self.v
                self.w+=self.alpha*delta*self.eligibility

            self.theta+=self.grad*self.beta
            # print(self.theta)
            # x=input()
            # if %100==0:
            # self.alpha-=self.alpha/(j+1)/10
                # if c%50==0:
                #     self.eps-=self.eps/c


            returns.append(reward)
            print(reward)
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
        b=(history[1]+0.07)/0.14;
        normalised_state=np.array([a,b])
        return normalised_state

   
    def run_sarsa(self,alpha,eps,t,n):
        r=[]
        for i in range(t):
            print('Sarsa trial',i)
            x=self.run_episode_sarsa(alpha,eps,n)
            # x=self.run_episode_sarsa_lambda(alpha,eps,n,0.2)
            print(x)
            r.append(x)
            # plt.plot(x)
            # plt.savefig('./mc/mcsarsa'+str(i)+str(alpha)+str(eps)+str(l)+'.jpg')
            # plt.clf()

        print('done')
        r=np.asarray(r)
        np.save('./mc/mcsarsa_trial'+str(i)+str(l)+'.npy',r)
        print(r.shape)
        vals=np.mean(r,axis=0)
        print(vals.shape)
        plt.plot(vals)
        plt.savefig('./mc/mcSarsa'+str(alpha)+str(eps)+str(l)+'.jpg')
        plt.clf()

        # plt.show()
        return vals

    def run_qlearn(self,alpha,eps,t,n):
        r=[]
        for i in range(t):
            print('Q trial',i)
            l=0
            # x=self.run_episode_q(alpha,eps,n)
            x=self.run_episode_q_lambda(alpha,eps,n,l)
            print(x)
            r.append(x)
            # plt.plot(x)
            # plt.savefig('./mc/mcq'+str(i)+str(alpha)+str(eps)+str(l)+'.jpg')
            # plt.clf()
            

        r=np.asarray(r)
        np.save('./mc/mcqlearn_trial'+str(i)+str(l)+'.npy',r)
        vals=np.mean(r,axis=0)
        print(vals.shape)
        plt.plot(vals)
        plt.savefig('./mc/mcQ'+str(alpha)+str(eps)+str(l)+'.jpg')
        plt.clf()
        return vals


    def run_ac(self,alpha,beta,l,t,n):
        r=[]
        for i in range(t):
            print('Q trial',i)
            l=0
            x=self.run_episode_ac(alpha,beta ,l,n)
            # x=self.run_episode_q_lambda(alpha,eps,n,0.2)
            print(x)
            r.append(x)
            # plt.plot(x)
            # plt.savefig('./mc/mcq'+str(i)+str(alpha)+str(eps)+'.jpg')
            # plt.clf()
            

        r=np.asarray(r)
        np.save('./acm/mcac_trial'+str(i)+str(l)+'.npy',r)
        vals=np.mean(r,axis=0)
        print(vals.shape)
        plt.plot(vals)
        plt.savefig('./acm/mcac_'+str(alpha)+str(beta)+str(l)+'.jpg')
        plt.clf()
        return vals

c=MountainCar()
# alpha=[1e-3,1e-2,1e-1,0.5,1]
# eps=[1,0.5,1e-2,1e-3]


# c.run_episode_ac(0.01,0.01,0.5,10)
c.run_ac(0.001,0.01,0.7,100,100)
# c.run_reinforce(0.001,0.01,0.1,50)
# 
# 
# alpha=[0.01]
# eps= [0.8]
# # # eps= [0.01]

# for a in alpha:
#     for e in eps:
#         print(a,e)
#         c.run_qlearn(a,e,100,100)
#         c.run_sarsa(a,e,100,100)