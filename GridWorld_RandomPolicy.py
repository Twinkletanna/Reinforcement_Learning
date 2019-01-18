# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

def attemp_action(state,selected_action):

    
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
    
    
def run_episode(gridworld,R):
    state=[0,0]
    sum=0
    i=0
    gamma=0.9
    count_34=0
    count_42=0
    count_both=0
    q4=True
    while state!=[4,4]:
        #actions=[0,1,2,3]
        #actions_names=['AU','AD','AR','AL']
        selected_action=np.random.choice(4,1)
        if q4==True:
            if i==8 and state==[3,4]:
                count_34=1
                #print('34')
            if i==19 and state==[4,2]:    
                count_42=1
                #print('42')
        new_state=attemp_action(state,selected_action)
        reward=R[new_state[0],new_state[1]]
        #if new_state==[4,2]:
        #    print(new_state,reward)
        state=new_state
        sum+=reward*gamma**i
        i+=1
        if reward not in (0,-10,10):
            print(reward)   
        
        if count_34==1 and count_42==1:
            count_both=1
        
    return sum,count_34,count_both
    
def run_optimal(gridworld,R,policy):
    state=[0,0]
    sum=0
    i=0
    gamma=0.9
    while state!=[4,4]:
        #actions=[0,1,2,3]
        #actions_names=['AU','AD','AR','AL']
        selected_action=policy[state[0]][state[1]]
        #print(state,selected_action)
        new_state=attemp_action(state,selected_action)
        reward=R[new_state[0],new_state[1]]
        #if new_state==[4,2]:
        #    print(new_state,reward)
        state=new_state
        sum+=reward*gamma**i
        if reward not in (0,-10,10):
            print(reward)
        i+=1
    return sum 

def q1(gridworld,reward):
    rewards=[]
    N=10000
    C12=0
    C1=0
    for i in range(0,N):
        sum,c1,c12=run_episode(gridworld,reward)
        rewards.append(sum)
        C12+=c12
        C1+=c1
    
    print(C12,C1)

    mean=np.mean(rewards)
    std=np.std(rewards)
    max=np.max(rewards)
    min=np.min(rewards)
    return(mean,std,min,max,C12/C1)    
    
def q3(gridworld,reward):
    
    policy=[[2,2,2,2,1],
            [2,2,2,2,1],
            [0,0,-1,1,1],
            [0,0,-1,1,1],
            [0,3,2,2,-1]]
    rewards=[]
    N=10000
    for i in range(0,N):
        rewards.append(run_optimal(gridworld,reward,policy))

    mean=np.mean(rewards)
    std=np.std(rewards)
    max=np.max(rewards)
    min=np.min(rewards)
    print('Q2,Q3: Optimal Policy')
    print('mean',mean,'std',std,'min',min,'max',max)
    

    
    
if __name__=="__main__":
    gridworld=np.zeros((5,5))
    gridworld[2][2]=1   
    gridworld[3][2]=1
    gridworld[4][2]=-1
    gridworld[4][4]=11
    reward=np.zeros((5,5))
    reward[4][2]=-10
    reward[4][4]=+10
    
    mean,std,min,max,prob=q1(gridworld,reward)
    print('Q1: Uniformly random Policy')
    print('mean',mean,'std',std,'min',min,'max',max)
    q3(gridworld,reward)
    
    
    probs=[]
    for i in range(0,10):
        _,_,_,_,prob=q1(gridworld,reward)
        probs.append(prob)
    print('Question 3. Experiment 1: ')
    print('Empirical probability:',np.mean(probs))
    
    
    
    
    

    