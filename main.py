#ZAFEIRAKIS KONSTANTINOS 2019030035
import numpy as np
import random
import math
import matplotlib.pyplot as plt

def max_div(a, b):
    """
    Selects the index of maximum a/b value from two arrays of equal length.

    Args:
        a (list): The first array.
        b (list): The second array.
    Returns:
        The index of the maximum a/b value.
    """
    max_division  = float('-inf') #initialize maximum value to -infinite
    max_index=-1                  #initialize maximum value index to -1
    for i in range(len(a)):
        if b[i] != 0:
            division = a[i] / b[i]
            if division > max_division:    #compare the current value to the previous max value
                max_division = division    #max division takes the highest value compared to the previous
                max_index=i                #max_index takes the index of the new highest value
    return max_index


def Setup(k):
    bandit_prob=[random.uniform(0,1) for i in range(k)]  # success probability for each arm
    bandit_rew = []                                      # reward for each arm

    for i in range(k):
        a = random.random()  # generate a random number between 0 and 1
        b = random.random()  # generate a random number between 0 and 1

        if a > b:
            a, b = b, a  # swap a and b if a is greater than b

        bandit_rew.append(random.uniform(a, b)) #each bandit arm takes a random value between the numbers a and b
    return bandit_prob , bandit_rew


def epsilon(k, T,bandit_prob,bandit_rew):
    CuArm = np.zeros((k,)) # cummulative reward for each arm
    inst_score = np.zeros((T,))  # reward for round t

    RealReward = np.zeros((T,))  # real reward of each arm mu
    best_score = np.zeros((T,))  # cumulative reward of best arm for round t
    alg_score = np.zeros((T,))  # cumulative reward for round t
    regret = np.zeros((T,))  # regret for round t
    pulls = np.zeros((k,))  # num of arm pulls
    CuRegret = np.zeros((T,)) #Cummulative Regret
    EpsilonComplexity = np.zeros((T,)) #Complexity for epsilon greedy theoretical

    for round in range(k):
        RealReward[round] = bandit_rew[round]*bandit_prob[round]  #find the average reward for each arm
        MaxRew=max(RealReward)                                    #select the maximum reward

    for round in range(T):

        epsilon = ((round+1)**(-1/3))*(k*math.log10(round+1))**((1/3)) #setting up the epsilon
        if np.random.random() < epsilon:                                                                #with probability epsilon : pick a random arm
            arm = np.random.randint(k)                                                                  #pick a random arm
            win_cond=np.random.binomial(1, p=bandit_prob[arm])
            CuArm[arm]=CuArm[arm] + bandit_rew[arm] * win_cond                                          #adding the reward to the cummulative reward per arm array
            pulls[arm]=pulls[arm]+1                                                                     #increase the number of arm pulls
            inst_score[round]=bandit_rew[arm] * win_cond                                                 #calculate the reward this round
        else:
            arm = max_div(CuArm,pulls)                                                                   #with probability 1-epsilon pick the best arm which is given by the max_div function
            win_cond = np.random.binomial(1, p=bandit_prob[arm]);
            CuArm[arm] = CuArm[arm] + bandit_rew[arm] * win_cond                                             #adding the reward to the cummulative reward per arm array
            pulls[arm] = pulls[arm] + 1                                                                     #increase the number of arm pulls
            inst_score[round]=bandit_rew[arm] * win_cond                                                     #calculate the reward this round

    #for max cummulative reward
        if round > 0:
            best_score[round] = best_score[round - 1] + MaxRew  # vector keeping track of t*optimal reward (cummulative reward)
        else:
            best_score[round] = MaxRew
    # for actual cummulative reward
        if round > 0:
            alg_score[round] = alg_score[round - 1] + inst_score[round]  # vector keeping track of cummulative reward at all times
        else:
            alg_score[round] = inst_score[round]
     # for regret
        regret[round] = (best_score[round] - alg_score[round]) / (round + 1)  # regret per iteration at round t
     # for  cummulative regret
        if round > 0:
            CuRegret[round] = CuRegret[round - 1] + regret[round]  # vector keeping track of cummulative regret at all times
        else:
            CuRegret[round] = regret[round]
     #for Epsilon complexity
        EpsilonComplexity[round] = ((round+1)**(2/3))*((k*math.log10(round+1))**(1/3))



    for round in range(k):
        print("Epsilon Greedy ",'Arm = %d: Reward = %f True probability = %f True mean = %f Sample mean = %f' % (round,bandit_rew[round],bandit_prob[round],RealReward[round],(CuArm[round])/pulls[round]))  #printing information about the arms

    #Plotting for Epsilon greedy
    fig, axs = plt.subplots(nrows=1, ncols=3)
    fig.suptitle("Epsilon-Greedy with k= %d , T=%d"% (k, T))
    axs[0].set_title("Performance")
    axs[0].set_xlabel("Round T")
    axs[0].set_ylabel("Reward")
    axs[0].plot(np.arange(1, T + 1), alg_score,label='Actual Reward' )
    axs[0].plot(np.arange(1, T + 1), best_score,label='Optimal (best arm) Reward')
    axs[0].legend()

    axs[1].set_title("Regret")
    axs[1].set_xlabel("Round T")
    axs[1].set_ylabel("Regret per round")
    axs[1].plot(np.arange(1, T + 1), regret)

    axs[2].set_title("Convergence rate")
    axs[2].set_xlabel("Round T")
    axs[2].set_ylabel("Cummulative Regret")
    axs[2].plot(np.arange(1, T + 1), CuRegret,label='Cummulative Regret')
    axs[2].plot(np.arange(1, T + 1), EpsilonComplexity,label='O(t^2/3 * (k*log(t))^1/3)')
    axs[2].legend()

    return regret #return the regret to plot it in the comparison graph
#-------------------------------------------------------------------------------------------------------------------UCB-------------------------------------------------------------------------------------------------------------------
def UCB(k,T,bandit_prob,bandit_rew):
    CuArm_UCB = np.zeros((k,)) # cummulative reward for each arm
    inst_score_UCB = np.zeros((T,))  # reward for round t

    RealReward = np.zeros((T,))  # real reward of each arm mu
    best_score = np.zeros((T,))  # cumulative reward of best arm for round t
    alg_score_UCB = np.zeros((T,))  # cumulative reward for round t
    regret_UCB = np.zeros((T,))  # regret for round t
    pulls_UCB = np.zeros((k,))  # num of arm pulls = QEstimate
    CuRegret_UCB=np.zeros((T,)) #Cummulative Regret

    UCBComplexity = np.zeros((T,)) #theoretical complexity
    MuEstimate = np.zeros((k,))
    UCB = [math.inf]*k #initialize at infinite


    for round in range(k):
        RealReward[round] = bandit_rew[round]*bandit_prob[round]  #find the average reward for each arm
        MaxRew=max(RealReward)                                    #select the maximum reward


    for round in range(T):

        arm= np.argmax(UCB)
        CuArm_UCB[arm] = CuArm_UCB[arm] + bandit_rew[arm] * np.random.binomial(1, p=bandit_prob[arm])  # adding the reward to the cummulative reward per arm array
        pulls_UCB[arm] = pulls_UCB[arm] + 1  # increase the number of arm pulls

        MuEstimate[arm] = CuArm_UCB[arm]/pulls_UCB[arm] #update the MuEstimate
        UCB[arm] = MuEstimate[arm] + math.sqrt(( math.log(T)/pulls_UCB[arm] ))   #update the UCB

        inst_score_UCB[round] = bandit_rew[arm] * np.random.binomial(1, p=bandit_prob[arm])  # calculate the reward this round

    # for max cummulative reward
        if round > 0:
            best_score[round] = best_score[round - 1] + MaxRew  # vector keeping track of t*optimal reward (cummulative reward)
        else:
            best_score[round] = MaxRew
    # for actual cummulative reward
        if round > 0:
            alg_score_UCB[round] = alg_score_UCB[round - 1] + inst_score_UCB[round]  # vector keeping track of cummulative reward at all times
        else:
            alg_score_UCB[round] = inst_score_UCB[round]
    #calculating regret
        regret_UCB[round] = (best_score[round] - alg_score_UCB[round]) / (round + 1)  # regret per iteration at round t
     # for  cummulative regret
        if round > 0:
            CuRegret_UCB[round] = CuRegret_UCB[round - 1] + regret_UCB[round]  # vector keeping track of cummulative regret at all times
        else:
            CuRegret_UCB[round] = CuRegret_UCB[round]
     # for Epsilon complexity
        UCBComplexity[round] = math.sqrt(k*(round+1)*math.log10(round+1))

    for round in range(k):
        print("UCB ",'Arm = %d: Reward = %f True probability = %f True mean = %f Sample mean = %f' % (round,bandit_rew[round],bandit_prob[round],RealReward[round],(CuArm_UCB[round])/pulls_UCB[round]))  #printing information about the arms


    #Plotting for UCB
    fig, axs = plt.subplots(nrows=1, ncols=3)
    fig.suptitle("UCB with k= %d , T=%d"% (k, T))

    axs[0].set_title("Performance")
    axs[0].set_xlabel("Round T")
    axs[0].set_ylabel("Reward")
    axs[0].plot(np.arange(1, T + 1), alg_score_UCB,label='Actual Reward' )
    axs[0].plot(np.arange(1, T + 1), best_score,label='Optimal (best arm) Reward')
    axs[0].legend()

    axs[1].set_title("Regret")
    axs[1].set_xlabel("Round T")
    axs[1].set_ylabel("Regret per round")
    axs[1].plot(np.arange(1, T + 1), regret_UCB)

    axs[2].set_title("Convergence rate")
    axs[2].set_xlabel("Round T")
    axs[2].set_ylabel("Regret per round")
    axs[2].plot(np.arange(1, T + 1), CuRegret_UCB,label='Cummulative Regret')
    axs[2].plot(np.arange(1, T + 1), UCBComplexity,label='O(sqrt(k*t*log(t))')
    axs[2].legend()

    return regret_UCB #return the regret to plot it in the comparison graph

#Plots the regret of epsilon greedy and UCB in the same plot
def plotter(a,b,k,T):
    fig, axs = plt.subplots(nrows=1, ncols=1)
    fig.suptitle("k= %d , T=%d"% (k, T))

    axs.set_title("Regret Comparison")
    axs.set_xlabel("Round T")
    axs.set_ylabel("Reward")
    axs.plot(np.arange(1, T + 1), a,label='Epsilon-Greedy' )
    axs.plot(np.arange(1, T + 1), b,label='UCB')
    axs.legend()

#________________________________________________RUNNING-TESTS___________________________________________________

#Senario1
prob , rew=Setup(10)
regret_e = epsilon(10,1000,prob,rew)
regret_UCB = UCB(10,1000,prob,rew)
plotter(regret_e,regret_UCB,10,1000)
#Senario2
prob , rew=Setup(100)
regret_e = epsilon(100,1000,prob,rew)
regret_UCB = UCB(100,1000,prob,rew)
plotter(regret_e,regret_UCB,100,1000)
#Senario3
prob , rew=Setup(10)
regret_e = epsilon(10,100000,prob,rew)
regret_UCB = UCB(10,100000,prob,rew)
plotter(regret_e,regret_UCB,10,100000)

plt.show()
