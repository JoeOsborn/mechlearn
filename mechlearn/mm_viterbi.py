import mm_systems
import numpy as np
import em

def viterbi(system):
    sigma = np.zeros([system.get_num_samples(), system.get_num_modes()])
    
    #compute sigma 0s
    sigma[0,:] = system.pm1 

    maxIs = np.ones([system.get_num_samples(),system.get_num_modes()])*-1

    #compute other sigmas
    for t in range(1, system.get_num_samples()):
        for j in range(0, system.get_num_modes()):
            max_sigmas = -1
            maxI = -1
            for i in range(0, system.get_num_modes()):
                sigmaI = 2*sigma[t-1, i]*em.guarded_trans_prob(t, i,j,system)*em.dynamics_lik(t, i, system)
                                
                if sigmaI > max_sigmas:
                    max_sigmas = sigmaI
                    maxI = i

            sigma[t, j] = max_sigmas
            maxIs[t, j] = maxI

    #backtrace and make assignments
    #Last assignment
    mT = np.argmax(sigma[-1,:])
    system.m[-1] = mT

    #Backtrace
    for t in range(-1, -system.get_num_samples(), -1):
        mT = maxIs[t, int(mT)] 
        system.m[t-1] = mT

    return system

#   Fills in the m data for the given system
#   Implements a modified version of log Viterbi algorithm from Rabiner(1988)
def log_viterbi(system):
    #Initialize sigma array
    sigma = np.zeros([system.get_num_samples(), system.get_num_modes()])
    
    #Compute sigmas for time step zero
    sigma[0,:] = np.log(system.pm1) #Rabiner 105a / Lane, Santana 13

    #Initialize the array to track the arguments that maximise the recursion step
    #This is used to backtrack below and get the most likely assignments
    maxIs = np.ones([system.get_num_samples(),system.get_num_modes()])*-1
    
    #Compute other sigmas recursively
    #For each remaining time step
    for t in range(1, system.get_num_samples()):
        #For each model j
        for j in range(0, system.get_num_modes()):
            max_sigmas = 0
            i = 0
            #Set the initial max sigma
            #This is necessary because it is possible for multiple models i to each have
            #zero transition probability and dynamics liklihood
            while max_sigmas == 0:
                trans_prob = em.guarded_trans_prob(t, i, j,system)
                if trans_prob != 0.0:
                    trans_prob = np.log(trans_prob)

                dynamics = em.dynamics_lik(t, i, system)
                if dynamics != 0.0:
                    dynamics = np.log(dynamics)

                max_sigmas = sigma[t-1, i] + trans_prob + dynamics
            maxI = i

            #Compute the value for all remaining values of i
            for i in range(i+1, system.get_num_modes()):
                trans_prob = em.guarded_trans_prob(t, i,j,system)
                if trans_prob != 0.0:
                    trans_prob = np.log(trans_prob)

                dynamics = em.dynamics_lik(t, i, system)
                if dynamics != 0.0:
                    dynamics = np.log(dynamics)

                sigmaI = sigma[t-1, i] + trans_prob + dynamics
                                
                if sigmaI > max_sigmas:
                    max_sigmas = sigmaI
                    maxI = i

            if max_sigmas == 0: #There are no non-zero sums for model j, simply set 
                                #sigma to the lowest possible integer value
                sigma[t, j] = -2147483648
            else:
                sigma[t, j] = max_sigmas
            maxIs[t, j] = maxI #record which mode i maximized sigma

    #backtrace and make assignments
    #The last mode is selected to be the mode that maximises sigma
    mT = np.argmax(sigma[-1,:])
    system.m[-1] = mT

    #Trace the path back from the final one
    for t in range(-1, -system.get_num_samples(), -1):
        mT = maxIs[t, int(mT)] 
        system.m[t-1] = mT

    return system
