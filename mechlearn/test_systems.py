import numpy as np
import mm_systems as mm
import em
import copy
import mm_svm
import matplotlib.pyplot as plt
from sklearn import svm
import math

def rc_circuit_em(Nsamples, sigma, sigmaLik, is_jmls, prob = -1.0, svm_mis_penalty=1000000):
    Ts = 1.0
    Rl=2000.0; Rc=400; C=0.3
    X0=3.5; Vin=10.0; Vmin=3.0; Vmax=4.0	
	
    #Data with correct mode assignments
    [data, noisy_data] = mm.rc_circuit(Nsamples,Ts,Rl,Rc,C,X0,Vin,Vmin,Vmax,sigma, sigmaLik,prob)
	
    #Same data, but with random mode assignments
    dataCopy = copy.deepcopy(data)	
    #dataCopy.m = np.random.binomial(1,0.5,noisy_data.m.size)
    for i in range(len(dataCopy.m)):
       	if(np.random.random() < 0.4):
            dataCopy.m[i] = np.random.randint(2, size = 1)   

 
    #Stores the experimental data
    system = mm.system_from_data(dataCopy)		

    K = 2

    #Estimates the activation regions using heavily corrupted data	
    if is_jmls:
        kernel_type='linear'

        classifier_list =  [ [] for i in range(K)]
        svm_data_x = np.array([[1.0, 10], [1.1,10],[1.3,10],[1.5,10],[1.7,10],[1.8,10],[2,10],[3.0,10],[3.2,10],[3.3,10],[3.5,10],[3.7,10],[3.9,10],[4.0,10]])
        svm_data_m = np.array([1,1,1,1,1,1,1,0,0,0,0,0,0,0])

        #Estimate SVMs
        for i in range(K):
            classifier_list[i] = classifier_list[i] = svm.SVC(kernel=kernel_type, C = svm_mis_penalty )
            classifier_list[i].fit(svm_data_x, svm_data_m)
    else:
        data_list = mm_svm.svm_split_data(dataCopy, system.K)
        classifier_list = mm_svm.svm_fit(data_list, system.K,mis_penalty=svm_mis_penalty)
			
    em.init_model(system,dataCopy,classifier_list,normalize_lik=True,scale_prob=True)
    [system, dataCopy, numSteps, avgExTime] = em.learn_model_from_data(system, dataCopy, data, is_jmls, svm_mis_penalty=svm_mis_penalty)
   
    simData = mm.simulate_mm_system(system)
    plt.figure()
    plt.plot(noisy_data.x[:,0],'b.',label='Training data')
    plt.plot(simData.x[:,0],'r',label='Sim. data')
    plt.legend()
    plt.show(block=False)

    return np.array([system, data, noisy_data, numSteps, avgExTime])

def rc_circuit_series_em():
    return np.array([system, data, noisy_data, numSteps, avgExTime])

def jet_route_fly_racetrack_em(nLoops, sigma, sigmaLik, is_jmls, svm_mis_penalty = 1000000):
    # Simulated data from the dynamical system, where we assume that
    # both x and u are observable.
	
    Ts = 0.2
	
    #Data with correct mode assignments
    [data, noisy_data] = mm.jet_fly_racetrack(nLoops, Ts, sigma, sigmaLik)
		
    #Same data, but with random mode assignments
    dataCopy = copy.deepcopy(data)	
    dataCopy.m = np.random.randint(3, size = data.m.size)
        
    #Stores the experimental data
    system = mm.system_from_data(dataCopy)
				
    K = 3 #True number of modes in the dataset	
    	
    if is_jmls:
        kernel_type='linear'
     
        classifier_list =  [ [] for i in range(K)]

        svm_data_x = np.ones((14, 4))
        svm_data_x[0, :] = -500
        svm_data_x[1,:] = -600
        svm_data_x[2,:] = -700
        svm_data_x[3,:] = -800
        svm_data_x[4,:] = -900
        svm_data_x[5,:] = -1000
        svm_data_x[6,:] = -1100
        svm_data_x[7,:] = 10
        svm_data_x[8,:] = -10
        svm_data_x[9,:] = 80
        svm_data_x[10,:] = -80
        svm_data_x[11,:] = 50   
        svm_data_x[12,:] = -50
        svm_data_x[13,:] = 35  
        svm_data_m = np.array([1,1,1,1,1,1,1,0,0,0,0,0,0,0])

        #Estimate SVMs
        for i in range(K):
            classifier_list[i] = classifier_list[i] = svm.SVC(kernel=kernel_type, C = svm_mis_penalty )
            classifier_list[i].fit(svm_data_x, svm_data_m)
    else:
        #Estimates the activation regions using heavily corrupted data	
        data_list = mm_svm.svm_split_data(dataCopy, K)
        classifier_list = mm_svm.svm_fit(data_list,K, svm_mis_penalty)
	
    em.init_model(system,dataCopy,classifier_list,normalize_lik=True,scale_prob=True)
    [system, dataCopy, numSteps, avgExTime] = em.learn_model_from_data(system, dataCopy,data, is_jmls, svm_mis_penalty=svm_mis_penalty)


    return np.array([system, data, noisy_data, numSteps, avgExTime])

def jet_route_fly_random_em(Nactions, sigma, sigmaLik, is_jmls, svm_mis_penalty=1000000):
    # Simulated data from the dynamical system, where we assume that
    # both x and u are observable.
	
    Ts = 0.2
	
    #Data with correct mode assignments
    [data, noisy_data] = mm.jet_fly_random(Nactions, Ts, sigma, sigmaLik)
		
    #Same data, but with random mode assignments
    dataCopy = copy.deepcopy(data)	
    #dataCopy.m = np.random.randint(3, size = data.m.size)
    for i in range(len(dataCopy.m)):
        if(np.random.random() < 0.4):
	    dataCopy.m[i] = np.random.randint(3, size = 1)
    
    #Stores the experimental data
    system = mm.system_from_data(dataCopy)
				
    K = 3 #True number of modes in the dataset	
    	
    if is_jmls:
        kernel_type='linear'

        classifier_list =  [ [] for i in range(K)]

        svm_data_x = np.ones((14, 4))
        svm_data_x[0, :] = -500
        svm_data_x[1,:] = -600
        svm_data_x[2,:] = -700
        svm_data_x[3,:] = -800
        svm_data_x[4,:] = -900
        svm_data_x[5,:] = -1000
        svm_data_x[6,:] = -1100
        svm_data_x[7,:] = 10
        svm_data_x[8,:] = -10
        svm_data_x[9,:] = 80
        svm_data_x[10,:] = -80
        svm_data_x[11,:] = 50   
        svm_data_x[12,:] = -50
        svm_data_x[13,:] = 35  
        svm_data_m = np.array([1,1,1,1,1,1,1,0,0,0,0,0,0,0])

        #Estimate SVMs
        for i in range(K):
            classifier_list[i] = classifier_list[i] = svm.SVC(kernel=kernel_type, C = svm_mis_penalty )
            classifier_list[i].fit(svm_data_x, svm_data_m)
    else:
        #Estimates the activation regions using heavily corrupted data	
        data_list = mm_svm.svm_split_data(dataCopy, K)
        classifier_list = mm_svm.svm_fit(data_list, K, mis_penalty = svm_mis_penalty)
	
    em.init_model(system,dataCopy,classifier_list,normalize_lik=True,scale_prob=True)
    [system, dataCopy, numSteps, avgExTime] = em.learn_model_from_data(system, dataCopy, data, is_jmls, svm_mis_penalty)
   
#    simData = mm.simulate_mm_system(system)
#    plt.figure()
#    plt.plot(data.x[:,0], data.x[:,2],'b.',label='True data')   
#    plt.plot(simData.x[:,0], simData.x[:,2],'r-',label='Sim. data') 
#    plt.legend()
#    plt.show(block=False)
#
    return np.array([system, data, noisy_data, numSteps, avgExTime])

def jet_route_fly_short_random_em(Nactions, sigma, sigmaLik, is_jmls, svm_mis_penalty=1000000):
    # Simulated data from the dynamical system, where we assume that
    # both x and u are observable.
	
    Ts = 0.2
	
    #Data with correct mode assignments
    [data, noisy_data] = mm.jet_fly_short_random(Nactions, Ts, sigma, sigmaLik)
		
    #Same data, but with random mode assignments
    dataCopy = copy.deepcopy(data)	
    #dataCopy.m = np.random.randint(3, size = data.m.size)
    for i in range(len(dataCopy.m)):
        if(np.random.random() < 0.4):
	    dataCopy.m[i] = np.random.randint(3, size = 1)
    
    #Stores the experimental data
    system = mm.system_from_data(dataCopy)
				
    K = 3 #True number of modes in the dataset	
    	
    if is_jmls:
        kernel_type='linear'

        classifier_list =  [ [] for i in range(K)]

        svm_data_x = np.ones((14, 4))
        svm_data_x[0, :] = -500
        svm_data_x[1,:] = -600
        svm_data_x[2,:] = -700
        svm_data_x[3,:] = -800
        svm_data_x[4,:] = -900
        svm_data_x[5,:] = -1000
        svm_data_x[6,:] = -1100
        svm_data_x[7,:] = 10
        svm_data_x[8,:] = -10
        svm_data_x[9,:] = 80
        svm_data_x[10,:] = -80
        svm_data_x[11,:] = 50   
        svm_data_x[12,:] = -50
        svm_data_x[13,:] = 35  
        svm_data_m = np.array([1,1,1,1,1,1,1,0,0,0,0,0,0,0])

        #Estimate SVMs
        for i in range(K):
            classifier_list[i] = classifier_list[i] = svm.SVC(kernel=kernel_type, C = svm_mis_penalty )
            classifier_list[i].fit(svm_data_x, svm_data_m)
    else:
        #Estimates the activation regions using heavily corrupted data	
        data_list = mm_svm.svm_split_data(dataCopy, K)
        classifier_list = mm_svm.svm_fit(data_list, K, mis_penalty = svm_mis_penalty)
	
    em.init_model(system,dataCopy,classifier_list,normalize_lik=True,scale_prob=True)
    [system, dataCopy, numSteps, avgExTime] = em.learn_model_from_data(system, dataCopy, data, is_jmls, svm_mis_penalty)
   
#    simData = mm.simulate_mm_system(system)
#    plt.figure()
#    plt.plot(data.x[:,0], data.x[:,2],'b.',label='True data')   
#    plt.plot(simData.x[:,0], simData.x[:,2],'r-',label='Sim. data') 
#    plt.legend()
#    plt.show(block=False)
#
    return np.array([system, data, noisy_data, numSteps, avgExTime])



def jet_route_fly_lawnmower_em(n_loops = 5, ymin = 0.0, ymax = 20.0, v0 = 2.0, Ts = 0.2, w = 5.0*math.pi/180.0, sigmaLik = 0.005, sigma = 0.05, is_jmls=False, svm_mis_penalty=1000000):
    # Simulated data from the dynamical system, where we assume that
    # both x and u are observable.
	
    #Data with correct mode assignments
    [data, noisy_data] = mm.jet_fly_lawnmower(n_loops, ymin, ymax, v0, Ts, w, sigmaLik, sigma)
		
    #Same data, but with random mode assignments
    dataCopy = copy.deepcopy(data)	
    #dataCopy.m = np.random.randint(3, size = data.m.size)
    for i in range(len(dataCopy.m)):
        if(np.random.random() < 0.4):
	    dataCopy.m[i] = np.random.randint(3, size = 1)
    
    #Stores the experimental data
    system = mm.system_from_data(dataCopy)
				
    K = 3 #True number of modes in the dataset	
    	
    if is_jmls:
        kernel_type='linear'

        classifier_list =  [ [] for i in range(K)]

        svm_data_x = np.ones((14, 4))
        svm_data_x[0, :] = -500
        svm_data_x[1,:] = -600
        svm_data_x[2,:] = -700
        svm_data_x[3,:] = -800
        svm_data_x[4,:] = -900
        svm_data_x[5,:] = -1000
        svm_data_x[6,:] = -1100
        svm_data_x[7,:] = 10
        svm_data_x[8,:] = 20
        svm_data_x[9,:] = 30
        svm_data_x[10,:] = 40
        svm_data_x[11,:] = 70   
        svm_data_x[12,:] = 90
        svm_data_x[13,:] = 100  
        svm_data_m = np.array([1,1,1,1,1,1,1,0,0,0,0,0,0,0])

        #Estimate SVMs
        for i in range(K):
            classifier_list[i] = classifier_list[i] = svm.SVC(kernel=kernel_type, C = svm_mis_penalty )
            classifier_list[i].fit(svm_data_x, svm_data_m)
    else:
        #Estimates the activation regions using heavily corrupted data	
        data_list = mm_svm.svm_split_data(dataCopy, K)
        classifier_list = mm_svm.svm_fit(data_list, K, mis_penalty = svm_mis_penalty)
	
    em.init_model(system,dataCopy,classifier_list,normalize_lik=True,scale_prob=True)
    [system, dataCopy, numSteps, avgExTime] = em.learn_model_from_data(system, dataCopy, data, is_jmls, svm_mis_penalty)
   
#    simData = mm.simulate_mm_system(system)
#    plt.figure()
#    plt.plot(data.x[:,0], data.x[:,2],'b.',label='True data')   
#    plt.plot(simData.x[:,0], simData.x[:,2],'r-',label='Sim. data') 
#    plt.legend()
#    plt.show(block=False)
#
    return np.array([system, data, noisy_data, numSteps, avgExTime])
