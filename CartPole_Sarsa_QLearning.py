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
   
class CartPole():
    """docstring for ClassName"""
    def __init__(self):
        self.state=()        
        self.time=0 
        self.state=(0,0,0,0)
        self.w=np.zeros((16,4))
        self.q=np.array((16,4))
        self.k=5
        self.gamma=1
        self.N=10    
        self.eps=1e-3

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
    

    def get_phi(self,state,k,z):
        c=list(comb(np.arange(k),repeat=4))
        c=np.asarray(c)
        s=self.normalise(state)
        phi=np.cos( math.pi * np.dot(c,s) ).ravel()
        phi_left= np.array([phi,z])
        phi_right=np.array([z,phi])

        return phi_left,phi_right

    def stop_cond(self,state):
        if self.state[2]>=math.pi/2 or self.state[2]<=-math.pi/2:
            return True
        if self.state[0]<=-3 or self.state[0]>=3:
            return True
        return False

    def egreedy(self,phi_left,phi_right,w):
        check=np.random.rand()
        if check<self.eps:
            a=np.random.choice([0,1],1)
            # print('random')
        else:
            # print('greedy')
            q_left=(self.w*phi_left).sum()
            q_right=(self.w*phi_right).sum()
            # print(self.w.shape,q_left.shape,q_right.shape)
            if q_left>q_right:
                a=0
            else:
                a=1

        if a==0:
            phi=phi_left
        elif a==1:
            phi=phi_right

        return a, phi

    def run_episode_sarsa(self,alpha,e,n):
        self.alpha=alpha
        self.eps=e
        k=6
        #change k
        # print('Fourier order:',k-1)
        
        self.w=np.zeros((2,k**4))
        z=np.zeros(k**4)
        returns=[]
        c=0
        for j in range(0,n):#200 eps
            self.state=(0,0,0,0)
            self.time=0
            hist=[]
            reward=0
            i=0

            # if j%10==0:
                # self.eps/=2

            phi_left,phi_right=self.get_phi(self.state,k,z)
            selected_action,this_phi=self.egreedy(phi_left,phi_right,self.w) 

            # selected_action=np.random.choice([0,1],1)
            # print(selected_action)

            while self.time<20.2  : 
                c+=1

                self.time+=0.02
                i+=1
                state=self.state
                self.state=self.take_action(self.state,selected_action)
                if self.stop_cond(state):
                    break
                reward+=1

                phi_left,phi_right=self.get_phi(state,k,z)
                next_phi_left,next_phi_right=self.get_phi(self.state,k,z)

                if selected_action==0:
                    this_phi=phi_left
                else:
                    this_phi=phi_right

                next_action, next_phi=self.egreedy(next_phi_left,next_phi_right,self.w)   


                #sarsa UPDATE
                self.q=(self.w*this_phi).sum()
                next_q=(self.w*next_phi).sum()
                delta=1+self.gamma*next_q-self.q
                # print(self.q.shape,next_q.shape,delta.shape)
                # print('update:',np.sum(delta))
                # print(delta)
                self.w+=self.alpha*this_phi*delta
                # print('done update')           
#                 if c %100==0:
#                     if self.eps>1e-3:
#                         self.eps-=self.eps/2
#                         # self.eps/=10
#                         # print('changed eps')
# # 
#                 if c%500==0:
#                     # alpha-=alpha/i
#                     if self.alpha>1e-8:
#                         self.alpha-=self.alpha/50
#                         # print('changed alpha')


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
        self.w=np.zeros((2,k**4))
        z=np.zeros(k**4)
        returns=[]
        c=0
        for j in range(0,n):#200 eps
            self.state=(0,0,0,0)
            self.time=0
            hist=[]
            reward=0
            i=0

            # if j%10==0:
                # self.eps/=2


            while self.time<20.2  : 
                c+=1
                self.time+=0.02
                i+=1



                phi_left,phi_right=self.get_phi(self.state,k,z)
                selected_action,this_phi=self.egreedy(phi_left,phi_right,self.w) 

                state=self.state
                self.state=self.take_action(self.state,selected_action)
                if self.stop_cond(state):
                    break

                # phi_left,phi_right=self.get_phi(state,k,z)
                next_phi_left,next_phi_right=self.get_phi(self.state,k,z)

                reward+=1

                if selected_action==0:
                    this_phi=phi_left
                else:
                    this_phi=phi_right

                # next_action, next_phi=self.egreedy(phi_left,phi_right,self.w)   

                #q UPDATE
                self.q=(self.w*this_phi).sum()
                next_q_left=(self.w*next_phi_left).sum()
                next_q_right=(self.w*next_phi_right).sum()
                # print(self.q.shape,next_phi_left.shape,next_q_left.shape)
                if next_q_left>next_q_right:
                    next_q=next_q_left

                else:
                    next_q=next_q_right


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


    def run_episode_poly_qlearn(self,alpha,e,n):
        self.alpha=alpha
        self.eps=e
        k=6
        self.w=np.zeros((2,k**4))
        z=np.zeros(k**4)
        returns=[]
        c=0
        for j in range(0,n):#200 eps
            self.state=(0,0,0,0)
            self.time=0
            hist=[]
            reward=0
            i=0
            while self.time<20.2  : 
                c+=1
                self.time+=0.02
                i+=1




                phi_left,phi_right=self.get_poly_phi(self.state,k,z)
                selected_action,this_phi=self.egreedy(phi_left,phi_right,self.w) 
                state=self.state
                self.state=self.take_action(self.state,selected_action)
                if self.stop_cond(state):
                    break
                next_phi_left,next_phi_right=self.get_poly_phi(self.state,k,z)

                reward+=1

                if selected_action==0:
                    this_phi=phi_left
                else:
                    this_phi=phi_right

                #q UPDATE
                self.q=(self.w*this_phi).sum()
                next_q_left=(self.w*next_phi_left).sum()
                next_q_right=(self.w*next_phi_right).sum()
                if next_q_left>next_q_right:
                    next_q=next_q_left

                else:
                    next_q=next_q_right


                delta=1+self.gamma*next_q-self.q
                self.w+=self.alpha*this_phi*delta

            returns.append(reward)
        return returns

    def run_episode_poly_sarsa(self,alpha,e,n):
        self.alpha=alpha
        self.eps=e
        k=6
        self.w=np.zeros((2,k**4))
        z=np.zeros(k**4)
        returns=[]
        c=0
        for j in range(0,n):#200 eps
            self.state=(0,0,0,0)
            self.time=0
            hist=[]
            reward=0
            i=0

            phi_left,phi_right=self.get_poly_phi(self.state,k,z)
            selected_action,this_phi=self.egreedy(phi_left,phi_right,self.w) 

            while self.time<20.2  : 
                c+=1

                self.time+=0.02
                i+=1
                state=self.state
                self.state=self.take_action(self.state,selected_action)
                if self.stop_cond(state):
                    break
                reward+=1

                phi_left,phi_right=self.get_poly_phi(state,k,z)
                next_phi_left,next_phi_right=self.get__poly_phi(self.state,k,z)

                if selected_action==0:
                    this_phi=phi_left
                else:
                    this_phi=phi_right

                next_action, next_phi=self.egreedy(next_phi_left,next_phi_right,self.w)   


                #sarsa UPDATE
                self.q=(self.w*this_phi).sum()
                next_q=(self.w*next_phi).sum()
                delta=1+self.gamma*next_q-self.q
                self.w+=self.alpha*this_phi*delta

                #END OF sarsa UPDATE

                selected_action=next_action

            returns.append(reward)
        return returns



    def normalise(self,history):
        a=(history[0]+3)*1.0/6;
        b=(history[1]+10)/20;
        c=(history[2]+math.pi/2)/(math.pi)
        d=(history[3]+math.pi/2)/(math.pi)
        normalised_state=np.array([a,b,c,d])
        return normalised_state

   
    def run_sarsa(self,alpha,eps,t,n):
        r=[]
        for i in range(t):
            print('Sarsa trial',i)
            x=self.run_episode_sarsa(alpha,eps,n)
            r.append(x)
            plt.plot(x)
            plt.savefig('./cpsarsa/cpsarsa'+str(i)+str(alpha)+str(eps)+'.jpg')
            plt.clf()
            np.save('./cpsarsa/cpsarsa_trial'+str(i)+'.npy',x)

        print('done')
        r=np.asarray(r)
        print(r.shape)
        vals=np.mean(r,axis=0)
        print(vals.shape)
        plt.plot(vals)
        plt.savefig('./CartpoleSarsa'+str(alpha)+str(eps)+'.jpg')
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
            plt.savefig('./cpq/cpq'+str(i)+str(alpha)+str(eps)+'.jpg')
            plt.clf()
            np.save('./cpq/cpqlearn_trial'+str(i)+'.npy',x)


        # print('done')
        r=np.asarray(r)
        # print(r.shape)
        vals=np.mean(r,axis=0)
        print(vals.shape)
        plt.plot(vals)
        plt.savefig('./CartpoleQ'+str(alpha)+str(eps)+'.jpg')
        plt.clf()
        # plt.show()
        return vals

    def run_poly_qlearn(self,alpha,eps,t,n):
        r=[]
        for i in range(t):
            print('Poly Qlearn trial',i)
            x=self.run_episode_poly_qlearn(alpha,eps,n)
            r.append(x)
            plt.plot(x)
            plt.savefig('./polyq/cppolyq'+str(i)+str(alpha)+str(eps)+'.jpg')
            plt.clf()
            np.save('./polyq/cppolyq_trial'+str(i)+'.npy',x)


        # print('done')
        r=np.asarray(r)
        # print(r.shape)
        vals=np.mean(r,axis=0)
        print(vals.shape)
        plt.plot(vals)
        plt.savefig('./CartpolePolyQ'+str(alpha)+str(eps)+'.jpg')
        plt.clf()
        # plt.show()
        return vals

    def run_poly_sarsa(self,alpha,eps,t,n):
        r=[]
        for i in range(t):
            print('Poly Sarsa trial',i)
            x=self.run_episode_poly_sarsa(alpha,eps,n)
            r.append(x)
            plt.plot(x)
            plt.savefig('./polys/cppolys'+str(i)+str(alpha)+str(eps)+'.jpg')
            plt.clf()
            np.save('./polys/cppolys_trial'+str(i)+'.npy',x)


        # print('done')
        r=np.asarray(r)
        # print(r.shape)
        vals=np.mean(r,axis=0)
        print(vals.shape)
        plt.plot(vals)
        plt.savefig('./CartpolePolyS'+str(alpha)+str(eps)+'.jpg')
        plt.clf()
        # plt.show()
        return vals


