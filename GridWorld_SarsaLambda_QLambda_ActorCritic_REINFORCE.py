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
        # self.q=np.random.rand(23,4)+10
        self.gamma=0.9
        # self.eps=0.001
        # self.eligibility=np.zeros(23)

        
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

    def new_egreedy(self,state_num):
        epsilon=self.eps
        a_star=np.argmax(self.q[state_num])
        weights = [1.0*epsilon/len(self.q[state_num])]*4 #One weight for each action
        weights[a_star]+=(1.0-self.eps)
        action = int(np.random.choice(4,1,p=weights)) #choose 1 in np.arange(4)
        return action

    def egreedy(self,state_num):
        check=np.random.rand()
        # print(self.eps)
        if check<self.eps:
            a=np.random.choice(4,1)
            # print('random')
            # x=input()
        else:
            # print(state_num)
            # print(self.q[state_num].shape)
            # print(self.q[state_num])
            a=np.argmax(self.q[state_num])
            # a=np.random.choice(4,1,p=self.q[state_num])
            # print('greedy')
            # print(a)
        # print(a)
        return a
    
    def sarsa_ep(self,state,alpha,eps):
        state=[0,0]
        self.eps=eps
        history=[state]
        sum=0
        i=0
        gamma=0.9

        state_num=self.grid[state[0]][state[1]]
        selected_action=self.egreedy(state_num)  
        selected_action=self.new_egreedy(state_num)  
        # alpha=0.05


        while state!=[4,4] and i<10000:
            # print(alpha,self.eps)

            new_state=self.attemp_action(state,selected_action)
            state_num=self.grid[state[0]][state[1]]
            new_state_num=self.grid[new_state[0]][new_state[1]]
            reward=self.reward[new_state[0],new_state[1]]
            # print(state,new_state,reward,selected_action)
            # x=input()
            next_action=self.egreedy(state_num)  
            next_action=self.new_egreedy(state_num)  

            # history.append(state)
            sum+=reward*gamma**i
            if reward not in (0,-10,10):
                print(reward)

            # if i%50==0 and i>0:
            #     alpha-=alpha/i
                # alpha=alpha/i
            i+=1

            delta=reward+self.gamma*self.q[new_state_num][next_action]-self.q[state_num][selected_action]
            self.q[state_num][selected_action]+=alpha*delta

            state=new_state
            selected_action=next_action
            self.alpha=alpha

        return sum , history

    
    # def sarsa_trial(self,alpha,eps,n):
    #     r=[]
    #     self.alpha=alpha
    #     self.q=np.random.rand(23,4)+10
    #     for i in range(0,n):
    #         returns,history=self.run_sarsa([0,0],alpha,eps)
    #         r.append(returns)
    #     return r


    def run_sarsa(self,alpha=0.5,eps=0.001,t=5,n=100,l=0.5):
        r=[]
        print('Gridworld')
        for i in range(t):
            print('trial',i)


            # x=self.sarsa_trial(alpha,eps,n)
            
            r_ep=[]
            self.alpha=alpha
            self.q=np.random.rand(23,4)+5
            for j in range(0,n):
                self.alpha=alpha
                # if self.eps>1e-4:
                    # self.eps-=self.eps/100
                ret,hist=self.sarsa_lambda_ep([0,0],self.alpha,eps)
                print(ret)
                r_ep.append(ret)

            r.append(r_ep)
            # plt.plot(r_ep)
            # plt.savefig('./gwgraph/sgw'+str(i)+str(alpha)+str(eps)+'.jpg')
            # plt.clf()
        
        np.save('./gwgraph/sgw_trial'+str(i)+str(alpha)+str(eps)+'.npy',r)
        print('done')
        r=np.asarray(r)
        print(r.shape)
        vals=np.mean(r,axis=0)
        print(vals.shape)
        plt.plot(vals)
        plt.savefig('./gwgraph/GridWorldSarsa'+str(alpha)+str(eps)+'.jpg')
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

        while state!=[4,4] and i<100:

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


    def run_qlearn(self,alpha,eps,t,n,l=0.5):
        all_trials=[]

        for j in range(t):
            print('trial',str(j))
            one_trial=[]
            self.q=np.random.rand(23,4)+5
            self.alpha=alpha
            self.eps=eps

            for i in range(n):
                r,hist=self.qlearn_lambda_ep([0,0],self.alpha,eps)
                print(r)
                one_trial.append(r)


            # plt.plot(one_trial)
            # plt.savefig('./gwgraph/qgw_trial'+str(j)+str(alpha)+str(l)+str(eps)+'.jpg')
            # plt.clf()

            all_trials.append(one_trial)

        np.save('./gwgraph/qgw_trial'+str(j)+str(alpha)+str(l)+str(eps)+'.npy',all_trials)
        avgd_trials=np.mean(all_trials,axis=0)
        plt.plot(avgd_trials)
        plt.savefig('./gwgraph/GridWorld_Qlearn'+str(alpha)+str(l)+str(eps)+'.jpg')
        plt.clf()

    def sarsa_lambda_ep(self,state,alpha,eps,l=0.5):
        state=[0,0]
        self.eligibility=np.zeros(23)
        self.eps=eps
        history=[state]
        sum=0
        i=0
        gamma=0.9

        state_num=self.grid[state[0]][state[1]]
        selected_action=self.egreedy(state_num)  
        # selected_action=self.new_egreedy(state_num)  
        # alpha=0.05


        while state!=[4,4] and i<1000:
            # print(alpha,self.eps)

            new_state=self.attemp_action(state,selected_action)
            state_num=self.grid[state[0]][state[1]]
            new_state_num=self.grid[new_state[0]][new_state[1]]
            reward=self.reward[new_state[0],new_state[1]]
            # print(state,new_state,reward,selected_action)
            # x=input()
            next_action=self.egreedy(state_num)  
            # next_action=self.new_egreedy(state_num)  

            self.eligibility=self.gamma*l*self.eligibility
            self.eligibility[state_num]+=1
            # print(self.eligibility)
            # x=input()

            # history.append(state)
            sum+=reward*gamma**i
            if reward not in (0,-10,10):
                print(reward)

            if i%25==0 and i>0:
                alpha-=alpha/i
                # alpha=alpha/i
            i+=1

            delta=reward+self.gamma*self.q[new_state_num][next_action]-self.q[state_num][selected_action]
            
            for each in range(0,self.q.shape[0]):
                self.q[each][selected_action]+=alpha*delta*self.eligibility[each]
            # self.q[state_num][selected_action]+=alpha*delta

            state=new_state
            selected_action=next_action
            self.alpha=alpha

        return sum , history

    def qlearn_lambda_ep(self,state,alpha,eps,l=0.5):
        state=[0,0]
        self.eligibility=np.zeros(23)
        self.eps=eps
        self.alpha=alpha
        history=[state]
        sum=0
        i=0
        gamma=0.9

        while state!=[4,4] and i<1000:

            state_num=self.grid[state[0]][state[1]]
            selected_action=self.egreedy(state_num)
            new_state=self.attemp_action(state,selected_action)
            new_state_num=self.grid[new_state[0]][new_state[1]]
            reward=self.reward[new_state[0],new_state[1]]
    

            self.eligibility=self.gamma*l*self.eligibility
            self.eligibility[state_num]+=1

            history.append(state)
            sum+=reward*gamma**i
            i+=1

            if i %25==0 and i>0:
                alpha-=alpha/i

            delta=reward+self.gamma*np.max(self.q[new_state_num])-self.q[state_num][selected_action]
            
            for each in range(0,self.q.shape[0]):
                self.q[each][selected_action]+=alpha*delta*self.eligibility[each]
            
            self.alpha=alpha
            state=new_state

        # print(state)
        return sum , history

    def softmax(self,state_num):
        self.temp=1
        m=np.max(self.theta[state_num])
        new_vec = math.e**((self.theta[state_num]-m)*self.temp)
        soft_q = new_vec/np.sum(new_vec) 
        a=int(np.random.choice(4,1,p=soft_q))
        # print(a)
        return a,soft_q[a]
    
    def softmax_reinforce(self,state_num):
        self.temp=1
        m=np.max(self.theta[state_num])
        new_vec = math.e**((self.theta[state_num]-m)*self.temp)
        soft_q = new_vec/np.sum(new_vec) 
        a=int(np.random.choice(4,1,p=soft_q))
        # print(a)
        return a,soft_q

    def AC(self,l):
        state=[0,0]
        self.Veligibility=np.zeros(23)
        self.Teligibility=np.zeros((23,4))

        history=[state]
        sum=0
        i=0
        self.gamma=0.9

        # print(self.theta)
        # x=input()

        while state!=[4,4] and i<1000:

            # Actor

            state_num=self.grid[state[0]][state[1]]
            selected_action,do=self.softmax(state_num)

            new_state=self.attemp_action(state,selected_action)
            new_state_num=self.grid[new_state[0]][new_state[1]]
            reward=self.reward[new_state[0],new_state[1]]
            history.append(state)
            sum+=reward*self.gamma**i
            i+=1

            #End actor

            
            #Critic Update
            self.Veligibility=self.gamma*l*self.Veligibility
            self.Veligibility[state_num]+=1
        
            delta=reward+self.gamma*self.v[new_state_num]-self.v[state_num]
            self.v+=self.alpha*delta*self.Veligibility
            #End critic update


            #Actor Update

            self.Teligibility=self.gamma*l*self.Teligibility
            self.Teligibility[state_num][selected_action]+=1-do
            self.theta+=self.beta*delta*self.Teligibility

            #End Actor Update


            state=new_state
            # print(state,selected_action,reward)

        return sum, history


    def reinforce(self,l):
        state=[0,0]
        self.eligibility=np.zeros(23)
        self.grad=np.zeros((23,4))
        history=[]
        sum=0
        i=0
        self.gamma=0.9

        # print(self.beta)
        # print(self.theta)
        # x=input()

        while state!=[4,4] and i<1000:

            # print(self.theta)
            # x=input()
            state_num=self.grid[state[0]][state[1]]
            selected_action,do=self.softmax_reinforce(state_num)

            new_state=self.attemp_action(state,selected_action)
            new_state_num=self.grid[new_state[0]][new_state[1]]
            reward=self.reward[new_state[0],new_state[1]]
            sum+=reward*self.gamma**i
            history.append([state_num,selected_action,new_state_num,do,sum,reward])
            i+=1
            state=new_state


        for j,each in enumerate(history):

            # print(each)
            state_num,selected_action,new_state_num,do,r,reward_t=each

            gt=(sum-r+reward_t)*(self.gamma**(-j))
            # self.grad[state_num]+=gt*-1*do
            # self.grad[state_num][selected_action]+=gt

            self.grad[state_num,:]+=(gt-self.v[state_num])*-1*do
            self.grad[state_num][selected_action]+=(gt-self.v[state_num])


            self.eligibility*=self.gamma*l
            self.eligibility[state_num]+=1           
            delta=r+self.gamma*self.v[new_state_num]-self.v[state_num]
            self.v+=self.alpha*delta*self.eligibility

            if j%50==0 and j!=0:
                self.beta-=self.beta/(j)
                self.alpha-=self.alpha/(j)
            
            # print(self.eligibility)
            # print(self.grad)

        

        # self.theta+=self.grad*self.alpha
        self.theta+=self.grad*self.beta
        # print(state_num,self.beta)
        # print('THETA',self.theta)
        # print('GRAD',self.grad)
        # x=input()
        # print(self.theta)
        # x=input()

        return sum, history


    def run_AC(self,alpha,beta,l,t,n):
        all_trials=[]

        for j in range(t):
            print('trial',str(j))
            one_trial=[]
            self.theta=np.zeros((23,4))
            self.v=np.zeros(23)
            self.alpha=alpha
            self.beta=beta
            for i in range(n):
                r,hist=self.AC(l)
                self.alpha-=self.alpha/(i+1)/10
                self.beta-=self.beta/(1+i)/10
                one_trial.append(r)

            print(one_trial[-10:])
            # plt.plot(one_trial)
            # plt.savefig('./ACgwgraph/ACgw_trial'+str(j)+str(alpha)+str(l)+str(eps)+'.jpg')
            # plt.clf()

            all_trials.append(one_trial)

        np.save('./ACgwgraph/ACgw_trial'+str(j)+str(alpha)+str(beta)+str(l)+'.npy',all_trials)
        avgd_trials=np.mean(all_trials,axis=0)
        plt.plot(avgd_trials)
        plt.grid(True)
        plt.title('Gridworld Actor Critic')
        plt.ylabel('Reward Averaged over 100 trials')
        plt.xlabel('Number of Episodes')
        ax = plt.subplot(1, 1, 1)
        ax.errorbar(x=[i for i in range(len(avgd_trials))], y=avgd_trials, yerr=np.std(avgd_trials), errorevery=10, color='green', ecolor='r',label='Actor Critic')
        plt.legend(loc='lower right')
        plt.savefig('./ACgwgraph/ACGridWorld_AC'+str(alpha)+str(beta)+str(l)+'.jpg')
        # plt.clf()


    def run_reinforce(self,alpha,beta,l,t,n):
        all_trials=[]

        for j in range(t):
            print('trial',str(j))
            one_trial=[]
            # self.theta=np.random.rand(23,4)
            self.theta=np.zeros((23,4))
            self.v=np.zeros(23)
            self.alpha=alpha
            self.beta=beta
            for i in range(n):
                r,hist=self.reinforce(l)
                # self.alpha-=self.alpha/(i+1)
                # print((i+1)*10)
                # self.beta-=self.beta/(1+i)
                one_trial.append(r)

            print(one_trial[-10:])
            # plt.plot(one_trial)
            # plt.savefig('./ACgwgraph/ACgw_trial'+str(j)+str(alpha)+str(l)+str(eps)+'.jpg')
            # plt.clf()
            
            all_trials.append(one_trial)

        np.save('./ACgwgraph/Rgw_trial'+str(j)+str(alpha)+str(beta)+str(l)+'.npy',all_trials)
        avgd_trials=np.mean(all_trials,axis=0)
        print(avgd_trials.shape)
        plt.plot(avgd_trials)
        plt.grid(True)
        plt.title('Gridworld REINFORCE')
        plt.ylabel('Reward Averaged over 100 trials')
        plt.xlabel('Number of Episodes')
        ax = plt.subplot(1, 1, 1)
        ax.errorbar(x=[i for i in range(len(avgd_trials))], y=avgd_trials, yerr=np.std(avgd_trials), errorevery=5, color='green', ecolor='r',label='REINFORCE')
        plt.legend(loc='lower right')
        plt.savefig('./ACgwgraph/RGridWorld_AC'+str(alpha)+str(beta)+str(l)+'.jpg')
        plt.clf()



g=GridWorld()
# g.run_sarsa(0.5,1,1,500)
# g.run_qlearn(0.5,0.001,1,100,0.5)


# lamda=[1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1]
# alpha=[1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1]
# eps=[1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1]

lamda=[0.9]
alpha=[0.5]
eps=[0.0001]

# sarsa, q lambda
# for a in alpha:
#     for e in eps:
#         for l in lamda:
#             print(a,e,l)
#             g.run_sarsa(a,e,100,100,l)
#             g.run_qlearn(a,e,100,100,l)


# sarsa q normal
# for a in alpha:
#     for e in eps:
#         print(a,e)
#         g.run_sarsa(a,e,100,100,0)
#         g.run_qlearn(a,e,100,100,0)



alpha=[0.001,0.01,0.1]
beta=[0.001,0.01,0.1]
lamda=[0.1,0.5,0.9]

# alpha=[0.0001,0.001,0.01,0.1]
# beta=[0.0001,0.001,0.01,0.1]
# lamda=[0.2,0.4,0.6,0.8,1]

#0.05 reached 1

alpha=[0.1]
beta=[0.1]
lamda=[0.1,0.2]


# AC FINAL PARAMS
# g.run_AC(0.1,0.2,0.4,100,250)

for a in alpha:
    for b in beta:
        for l in lamda:
            print(a,b,l)
            g.run_reinforce(a,b,l,100,100)
