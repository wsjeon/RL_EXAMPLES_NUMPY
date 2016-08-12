# ============================================================================
# Wonseok Jeon, EE, KAIST
# 2016/08/12: Epsilon greedy
# ============================================================================
import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt

# (0) Parameters
gamma = 0.99 # discount factor
alpha = 0.1 # learning rate
mem_size = 1 # experience memory size
K = 1 # number of Q headers
prob = 1  # probability to update each Q
replay = 0 # replay or not
STDDEV = 0 # standard deviation for randomization
target_epsilon = 0

# (1) Q values: 10 states, 2 actions, K headers
def Initialize_Q(STDDEV, K):
    if STDDEV == 0:
        Q = np.zeros([10, 2, K])
    else:
        Q = np.random.normal(0, STDDEV, [10, 2, K])
    return Q

# (2) Q update function
# ----- Q: Q table
# ----- S: State (Current)
# ----- A: Action
# ----- R: Reward
# ----- Sn: State (Next)
# ----- M: Mask
def updater(Q, S, A, R, Sn, M):
    for k in np.where(M == 1)[0]:
        Target = R + gamma * Q[Sn, :, k].max()
        TD = Target - Q[S, A, k]
        Q[S, A, k] = Q[S, A, k] + alpha * TD    

# (3) Replay memory update function
# ----- MEM: replay memory
# ----- S: State (Current)
# ----- A: Action 
# ----- R: Reward 
# ----- Sn: State (Next)
MEM = np.zeros([4+K, mem_size])

def memoryIN(MEM, S, A, R, Sn):
    MEM = np.roll(MEM, 1, axis=1) 
    MEM[0, 0] = S
    MEM[1, 0] = A
    MEM[2, 0] = R
    MEM[3, 0] = Sn
    MEM[4:, 0] = np.random.binomial(1, prob, K)

    return MEM
def EpsGreedyPolicy(epsilon, Q, k, St):
    if np.random.uniform()>epsilon:
        if Q[St, 0, k] > Q[St, 1, k]:
            At = 0
        elif Q[St, 0, k] < Q[St, 1, k]:
            At = 1
        else:
            At = np.random.randint(0, 2)
    else:
        At = np.random.randint(0, 2)
    return At
# (4) Environment
# ----- S: State (Current)
# ----- A: Action
# ----- R: Reward 
# ----- Sn: State (Next)
def env1(S, A):
    if A == 0:
        Sn = S - 1
    else:
        Sn = S + 1
    if Sn == 0:
        return 1, 1
    elif Sn == 9:
        return 1000, 1
    else:
        return 0, Sn

def env3(S, A):
    if A == 0:
        Sn = S - 1
    else:
        Sn = S + 1
    if Sn < 0 or Sn > 9:
        Sn = S

    if Sn == 0:
        return 1, Sn
    elif Sn == 9:
        return 100, Sn
    else:
        return 0, Sn

# (5) Desired Result
max_ann_time = 4001
step_ann_time = 400
init_ann_time =3200

max_run_time = 100
step_run_time = 1
init_run_time = 0

max_tm = 5000
step_tm = 1
init_tm = 0

test_tm = 100

Test_Reward = np.zeros([max_run_time,max_tm/test_tm,max_ann_time/step_ann_time]) 

for annealing_time in range(init_ann_time,max_ann_time,step_ann_time): # Annealing time 
    for run_time in range(init_run_time,max_run_time,step_run_time):
        # ==================================================
        # Run the game
        # ==================================================
        St = 1 # Initialize the state
        Q = Initialize_Q(STDDEV, K) # Initialize Q
        
        for time_step in range(init_tm,max_tm,step_tm): # 1000 time steps
            start_time = time.time()
            if time_step < annealing_time:
                epsilon = 1 - (1 - target_epsilon) * float(time_step) / float(annealing_time)
            else:
                epsilon = target_epsilon
            
            k = np.random.randint(K) # Choosen header for this episode
            # 1) Select action using epsilon greedy policy 
            At = EpsGreedyPolicy(epsilon, Q, k, St)
                
            # 2) Do action & Get next state and reward.
            Rn, Sn = env1(St, At)
              
            # 3) Memory In
            MEM = memoryIN(MEM, St, At, Rn, Sn)
            
            # 4) Update Q value by using the memory
            updater(Q, MEM[0, 0], MEM[1, 0], MEM[2, 0], MEM[3, 0], MEM[4:, 0])
    
            # 5) State Transition
            os.system('clear')                
    
            print "============================================================="
            print "          Q Learning via Epsilon Greedy Policy"
            print "                 (Q%d is currently used)" % k
            print "   Annealing:",annealing_time,", Time Step:",time_step, ", Run:", run_time
            print (time.time() - start_time), "seconds"
            print "============================================================="
            print "curr State :", St
            if At == 0:
                print "Action: <<<<----------"
            else:
                print "Action: ---------->>>>"
            print "Reward:", Rn
            print "next State:", Sn
            print "Epsilon:", epsilon
            print "Q values:"
            print Q[:, :, k]
            
            # 5) State Transition
            St = Sn
            
            # 6) Test
            if time_step % test_tm == 0:
                Stest = 1 
                Test_Reward_Tmp = 0
                for test_time_step in range(0,10,1):
                    Atest = EpsGreedyPolicy(0, Q, k, Stest)
                    Rntest, Sntest = env1(Stest, Atest)
                    Test_Reward_Tmp = Test_Reward_Tmp + Rntest
                    Stest = Sntest
                    
                Test_Reward[run_time, time_step/test_tm, annealing_time/step_ann_time-1] = Test_Reward_Tmp
                print Test_Reward_Tmp

        STR0 = '_GM_'+str(gamma)
        STR1 = '_LR_'+str(alpha)
        STR2 = '_K_'+str(K)
        STR3 = '_SD_'+str(STDDEV)
        STR4 = '_TE_'+str(target_epsilon)   
        np.save('DATA_'+STR0+STR1+STR2+STR3+STR4+'.npy', Test_Reward)


