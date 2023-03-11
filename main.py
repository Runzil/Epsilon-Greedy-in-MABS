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


k = 10  # number of arms
T = 1000  # horizon


bandit_prob=[random.uniform(0,1) for i in range(k)]  # success probability for each arm
bandit_rew = []                                      # reward for each arm

for i in range(k):
    a = random.random()  # generate a random number between 0 and 1
    b = random.random()  # generate a random number between 0 and 1

    if a > b:
        a, b = b, a  # swap a and b if a is greater than b

    bandit_rew.append(random.uniform(a, b)) #each bandit arm takes a random value between the numbers a and b

# print("probabilities",bandit_prob)
# print("rewards",bandit_rew)


CuArm = np.zeros((k,)) # cummulative reward for each arm
inst_score = np.zeros((T,))  # reward for round t

RealReward = np.zeros((T,))  # real reward of each arm mu
best_score = np.zeros((T,))  # cumulative reward of best arm for round t
alg_score = np.zeros((T,))  # cumulative reward for round t
regret = np.zeros((T,))  # regret for round t
pulls = np.zeros((k,))  # num of arm pulls


for round in range(k):
    RealReward[round] = bandit_rew[round]*bandit_prob[round]  #find the average reward for each arm
    # print('Arm = %d: Reward = %f True probability = %f True mean = %f ' % (round,bandit_rew[round],bandit_prob[round],RealReward[round]))  #printing information about the arms
    MaxRew=max(RealReward)                                    #select the maximum reward

for round in range(T):

    epsilon = ((round+1)**(-1/3))*(math.log10(round+1))**((1/3)) #setting up the epsilon
    #epsilon=0.9
    if np.random.random() < epsilon:                                                                #with probability epsilon : pick a random arm
        arm = np.random.randint(k)                                                                  #pick a random arm
        CuArm[arm]=CuArm[arm] + bandit_rew[arm] * np.random.binomial(1, p=bandit_prob[arm])         #adding the reward to the cummulative reward per arm array
        pulls[arm]=pulls[arm]+1                                                                     #increase the number of arm pulls
        inst_score[round]=bandit_rew[arm] * np.random.binomial(1, p=bandit_prob[arm])               #calculate the reward this round
    else:
        arm = max_div(CuArm,pulls)                                                                   #with probability 1-epsilon pick the best arm which is given by the max_div function
        CuArm[arm] = CuArm[arm] + bandit_rew[arm] * np.random.binomial(1, p=bandit_prob[arm])        #adding the reward to the cummulative reward per arm array
        pulls[arm] = pulls[arm] + 1                                                                     #increase the number of arm pulls
        inst_score[round]=bandit_rew[arm] * np.random.binomial(1, p=bandit_prob[arm])                   #calculate the reward this round

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
    regret[round] = (best_score[round] - alg_score[round]) / (round + 1)  # regret per iteration at round t


for round in range(k):
    print('Arm = %d: Reward = %f True probability = %f True mean = %f Sample mean = %f' % (round,bandit_rew[round],bandit_prob[round],RealReward[round],(CuArm[round])/pulls[round]))  #printing information about the arms


#Plotting
fig, axs = plt.subplots(nrows=1, ncols=2)

axs[0].set_title("Epsilon-Greedy Performance")
axs[0].set_xlabel("Round T")
axs[0].set_ylabel("Reward")
axs[0].plot(np.arange(1, T + 1), alg_score,label='Actual Reward' )
axs[0].plot(np.arange(1, T + 1), best_score,label='Optimal (best arm) Reward')
axs[0].legend()

axs[1].set_title("Epsilon-Greedy Regret")
axs[1].set_xlabel("Round T")
axs[1].set_ylabel("Regret per round")
axs[1].plot(np.arange(1, T + 1), regret)
plt.show()
