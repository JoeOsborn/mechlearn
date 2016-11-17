import testing
import mm_systems as mm
import copy
import numpy as np
import linear_model_fit as lmf
from sklearn import svm
import em
import time
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt

def jmls():
    Nsamples = 700
    Ts = 1.0
    Rl=2000.0; Rc=400; C=0.3
    X0=3.5; Vin=10.0; Vmin=3.0; Vmax=4.0

    #Data with correct mode assignments
    data = mm.rc_circuit(Nsamples,Ts,Rl,Rc,C,X0,Vin,Vmin,Vmax,prob=-1.0)

    #Same data, but with random mode assignments
    dataCopy = copy.deepcopy(data)
    dataCopy.m = np.random.binomial(1,0.5,data.m.size)

    system = mm.system_from_data(dataCopy)
    all_datasets = lmf.split_datasets(dataCopy)
    datasets_per_model = lmf.group_models(all_datasets)
    K = len(datasets_per_model)

    kernel_type='linear'
    mis_penalty = 1000000


    classifier_list =  [ [] for i in range(K)]
    svm_data_x = np.array([[-0.5, -10], [-0.5,-10],[-0.5,-10],[-0.5,-10],[-0.5,-10],[-0.5,-10],[-0.5,-10],[0.5,-10],[0.5,-10],[0.5,-10],[0.5,-10],[0.5,-10],[0.5,-10],[0.5,-10]])
    svm_data_m = np.array([-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1])

    #Estimate SVMs
    for i in range(K):
        classifier_list[i] = classifier_list[i] = svm.SVC(kernel=kernel_type, C = mis_penalty, class_weight={1: 1})
        classifier_list[i].fit(svm_data_x, svm_data_m)

    for i in range(K):
        [A,B]=lmf.linear_model_fit(datasets_per_model[i])
        system.Am.append(A)
        print "Model "+str(i)+": A is \n"+str(A)
        if B != None:
            system.Bm.append(B)
            print "Model "+str(i)+": B is \n"+str(B)+"\n"

    for i in range(system.K):
        system.Am[i]+= np.random.rand(1)*0.001
        system.Bm[i]+= np.random.rand(1)*0.001


    simDataDisturbed = mm.simulate_mm_system(system)
    em.init_model(system,classifier_list,normalize_lik=True,scale_prob=True)
    max_steps = 20

    avg_likelihood = np.array([em.avg_log_likelihood(system)])
    mis_models = np.array([em.count_different(dataCopy.m[:-1],data.m[:-1])])

    for i in range(max_steps):
        print "\nStep "+str(i)+""
        #E step 
        print "E step..."
        st = time.clock()
        em.compute_alphas(system)
        em.compute_betas(system)
        em.compute_gamma(system)
        em.compute_xi(system)

        st = time.clock()
        #Estimates the most likely labels
        em.approx_viterbi(system)
        dataCopy.m = np.copy(system.m)

        print "Elapsed time [s]: "+str(time.clock()-st)+""
        print "M step..."
        st = time.clock()
        em.max_initial_mode_probs(system)
        em.max_linear_models(system)
        em.max_guarded_trans_probs(system)
        print "Elapsed time [s]: "+str(time.clock()-st)+""
    
        avg_likelihood = np.append(avg_likelihood,em.avg_log_likelihood(system))
        mis_models = np.append(mis_models,em.count_different(dataCopy.m[:-1],data.m[:-1]))

        print "Average log-likelihood: "+str(avg_likelihood[-1])
        print "Misclassified models: "+str(mis_models[-1])

        if np.abs(avg_likelihood[-2]-avg_likelihood[-1])<5:
            print "\n\n********* Model converged *********\n\n"
            break

    simDataEM = mm.simulate_mm_system(system)

    initial_rmse = em.rmse(data.x,simDataDisturbed.x)
    final_rmse = em.rmse(data.x,simDataEM.x)
       
    print "\n\n***** Initial RMSE= "+str(initial_rmse)
    print "***** Final RMSE= "+str(final_rmse)+"\n\n"

    print "***** Initial mislabeled points= "+str(mis_models[0])
    print "***** Final mislabeled points= "+str(mis_models[-1])+"\n\n"

    plt.figure()
    plt.plot(data.x[:,0],'b.',label='True data')
    plt.plot(simDataEM.x[:,0],'r-',label='JMLS Sim. data')
    plt.grid()
    plt.legend()
    plt.title("Model fit")
    plt.show(block=False)


    return final_rmse

def jmls_rc_circuit(Nsamples):
    Ts = 1.0
    Rl=2000.0; Rc=400; C=0.3
    X0=3.5; Vin=10.0; Vmin=3.0; Vmax=4.0; sigma = 0.05

    #Data with correct mode assignments
    [data, noisy_data] = mm.rc_circuit(Nsamples,Ts,Rl,Rc,C,X0,Vin,Vmin,Vmax,sigma,prob=-1.0)

    #Same data, but with random mode assignments
    dataCopy = copy.deepcopy(noisy_data)

    #Same data, but with random mode assignments
    dataCopy = copy.deepcopy(noisy_data)
    dataCopy.m = np.random.binomial(1,0.5,data.m.size)

    system = mm.system_from_data(dataCopy)
    all_datasets = lmf.split_datasets(dataCopy)
    datasets_per_model = lmf.group_models(all_datasets)
    K = len(datasets_per_model)

    kernel_type='linear'
    mis_penalty = 1000000


    classifier_list =  [ [] for i in range(K)]
    svm_data_x = np.array([[1.0, 10], [1.1,10],[1.3,10],[1.5,10],[1.7,10],[1.8,10],[2,10],[3.0,10],[3.2,10],[3.3,10],[3.5,10],[3.7,10],[3.9,10],[4.0,10]])
    svm_data_m = np.array([1,1,1,1,1,1,1,0,0,0,0,0,0,0])

    #Estimate SVMs
    for i in range(K):
        classifier_list[i] = classifier_list[i] = svm.SVC(kernel=kernel_type, C = mis_penalty )
        classifier_list[i].fit(svm_data_x, svm_data_m)

    for i in range(K):
        [A,B]=lmf.linear_model_fit(datasets_per_model[i])
        system.Am.append(A)
        print "Model "+str(i)+": A is \n"+str(A)
        if B != None:
            system.Bm.append(B)
            print "Model "+str(i)+": B is \n"+str(B)+"\n"

    for i in range(system.K):
        system.Am[i]+= np.random.rand(1)*0.007
        system.Bm[i]+= np.random.rand(1)*0.007


    em.init_model(system,classifier_list,normalize_lik=True,scale_prob=True)
    max_steps = 20

    avg_likelihood = np.array([em.avg_log_likelihood(system)])
    mis_models = np.array([em.count_different(dataCopy.m[:-1],data.m[:-1])])

    numSteps = 0
    avgExTime = 0
    conv_count = 0

    for i in range(max_steps):
        print "\nStep "+str(i)+""
        #E step 
        print "E step..."
        st = time.clock()
        em.compute_alphas(system)
        em.compute_betas(system)
        em.compute_gamma(system)
        em.compute_xi(system)

        #Estimates the most likely labels
        em.approx_viterbi(system)
        dataCopy.m = np.copy(system.m)

        
        print "M step..."
        em.max_initial_mode_probs(system)
        em.max_linear_models(system)
        em.max_guarded_trans_probs(system)
        avgExTime += time.clock()-st
    
        avg_likelihood = np.append(avg_likelihood,em.avg_log_likelihood(system))
        mis_models = np.append(mis_models,em.count_different(dataCopy.m[:-1],data.m[:-1]))

        print "Average log-likelihood: "+str(avg_likelihood[-1])
        print "Misclassified models: "+str(mis_models[-1])
        
        numSteps += 1
        if np.abs(avg_likelihood[-2]-avg_likelihood[-1])<5:
            conv_count+=1
        else:
            conv_count = 0


        if conv_count >= 15:
            print "\n\n********* Model converged *********\n\n"
            break


    avgExTime = avgExTime/numSteps

    simDataEM = mm.simulate_mm_system(system)
    plt.figure()
    plt.plot(noisy_data.x[:,0],'b.',label='True data')
    plt.plot(simDataEM.x[:,0],'r-',label='JMLS Sim. data')
    plt.grid()
    plt.legend()
    plt.title("Model fit")
    plt.show(block=False)


    return np.array([system, data, noisy_data, numSteps, avgExTime])

def jmls_rc_circuit_series(Nsamples, Ncircuits):
    Ts = 1.0
    Rl=2000.0; Rc=400; C=0.3
    X0=3.5; Vin=10.0; Vmin=3.0; Vmax=4.0; sigma = 0.007

    #Data with correct mode assignments
    data = mm.rc_circuit_series(Ncircuits,Nsamples,Ts,Rl,Rc,C,X0,Vin,Vmin,Vmax,sigma,prob=-1.0)

    #Same data, but with random mode assignments
    dataCopy = copy.deepcopy(data)

    #Same data, but with random mode assignments
    dataCopy = copy.deepcopy(data)
    dataCopy.m = np.random.binomial(1,0.5,data.m.size)

    system = mm.system_from_data(dataCopy)
    all_datasets = lmf.split_datasets(dataCopy)
    datasets_per_model = lmf.group_models(all_datasets)
    K = len(datasets_per_model)

    kernel_type='linear'
    mis_penalty = 1000000


    classifier_list =  [ [] for i in range(K)]

    svm_data_x = np.ones((12, nCircuits + 1))
    svm_data_x[0,:] = 1.0
    svm_data_x[1,:] = 1.3
    svm_data_x[2,:] = 1.5
    svm_data_x[3,:] = 1.7
    svm_data_x[4,:] = 1.8
    svm_data_x[5,:] = 2.0
    svm_data_x[6,:] = 3.0
    svm_data_x[7,:] = 3.2
    svm_data_x[8,:] = 3.3
    svm_data_x[9,:] = 3.5
    svm_data_x[10,:] = 3.7
    svm_data_x[11,:] = 4.0
    svm_data_x[:,-1] = 10   
    svm_data_m = np.array([-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1])

    #Estimate SVMs
    for i in range(K):
        classifier_list[i] = classifier_list[i] = svm.SVC(kernel=kernel_type, C = mis_penalty )
        classifier_list[i].fit(svm_data_x, svm_data_m)

    for i in range(K):
        [A,B]=lmf.linear_model_fit(datasets_per_model[i])
        system.Am.append(A)
        print "Model "+str(i)+": A is \n"+str(A)
        if B != None:
            system.Bm.append(B)
            print "Model "+str(i)+": B is \n"+str(B)+"\n"

    for i in range(system.K):
        system.Am[i]+= np.random.rand(1)*0.001
        system.Bm[i]+= np.random.rand(1)*0.001


    em.init_model(system,classifier_list,normalize_lik=True,scale_prob=True)
    max_steps = 20

    avg_likelihood = np.array([em.avg_log_likelihood(system)])
    mis_models = np.array([em.count_different(dataCopy.m[:-1],data.m[:-1])])

    numSteps = 0
    avgExTime = 0

    for i in range(max_steps):
        print "\nStep "+str(i)+""
        #E step 
        print "E step..."
        st = time.clock()
        em.compute_alphas(system)
        em.compute_betas(system)
        em.compute_gamma(system)
        em.compute_xi(system)

        #Estimates the most likely labels
        em.approx_viterbi(system)
        dataCopy.m = np.copy(system.m)

        
        print "M step..."
        em.max_initial_mode_probs(system)
        em.max_linear_models(system)
        em.max_guarded_trans_probs(system)
        avgExTime += time.clock()-st
    
        avg_likelihood = np.append(avg_likelihood,em.avg_log_likelihood(system))
        mis_models = np.append(mis_models,em.count_different(dataCopy.m[:-1],data.m[:-1]))

        print "Average log-likelihood: "+str(avg_likelihood[-1])
        print "Misclassified models: "+str(mis_models[-1])
        
        numSteps += 1
        if np.abs(avg_likelihood[-2]-avg_likelihood[-1])<5:
            print "\n\n********* Model converged *********\n\n"
            break

    avgExTime = avgExTime/numSteps

    simDataEM = mm.simulate_mm_system(system)
    plt.figure()
    plt.plot(data.x[:,0],'b.',label='True data')
    plt.plot(simDataEM.x[:,0],'r-',label='JMLS Sim. data')
    plt.grid()
    plt.legend()
    plt.title("Model fit")
    plt.show(block=False)


    return np.array([system, data, numSteps, avgExTime])

def jmls_fly_racetrack(Nloops):
    Ts = 1.0
    sigma = 0.1

    #Data with correct mode assignments
    [data, noisy_data] = mm.jet_fly_racetrack(Nloops, Ts, sigma)

    #Same data, but with random mode assignments
    dataCopy = copy.deepcopy(noisy_data)
    dataCopy.m = np.random.randint(3, size = data.m.size)

    system = mm.system_from_data(dataCopy)
    all_datasets = lmf.split_datasets(dataCopy)
    datasets_per_model = lmf.group_models(all_datasets)
    K = len(datasets_per_model)

    kernel_type='linear'
    mis_penalty = 1000000


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
        classifier_list[i] = classifier_list[i] = svm.SVC(kernel=kernel_type, C = mis_penalty )
        classifier_list[i].fit(svm_data_x, svm_data_m)

    for i in range(K):
        [A,B]=lmf.linear_model_fit(datasets_per_model[i])
        system.Am.append(A)
        print "Model "+str(i)+": A is \n"+str(A)

    for i in range(system.K):
        system.Am[i]+= np.random.rand(1)*0.001


    em.init_model(system,classifier_list,normalize_lik=True,scale_prob=True)
    max_steps = 200

    avg_likelihood = np.array([em.avg_log_likelihood(system)])
    mis_models = np.array([em.count_different(dataCopy.m[:-1],data.m[:-1])])

    numSteps = 0
    avgExTime = 0

    conv_count = 0

    for i in range(max_steps):
        print "\nStep "+str(i)+""
        #E step 
        print "E step..."
        st = time.clock()
        em.compute_alphas(system)
        em.compute_betas(system)
        em.compute_gamma(system)
        em.compute_xi(system)

        #Estimates the most likely labels
        em.approx_viterbi(system)
        dataCopy.m = np.copy(system.m)

        
        print "M step..."
        em.max_initial_mode_probs(system)
        em.max_linear_models(system)
        em.max_guarded_trans_probs(system)
        avgExTime += time.clock()-st
    
        avg_likelihood = np.append(avg_likelihood,em.avg_log_likelihood(system))
        mis_models = np.append(mis_models,em.count_different(dataCopy.m[:-1],data.m[:-1]))

        print "Average log-likelihood: "+str(avg_likelihood[-1])
        print "Misclassified models: "+str(mis_models[-1])
        
        numSteps += 1
        if np.abs(avg_likelihood[-2]-avg_likelihood[-1])<3:
            conv_count+=1
        else:
            conv_count = 0


        if conv_count >= 15:
            print "\n\n********* Model converged *********\n\n"
            break


    avgExTime = avgExTime/numSteps

    simDataEM = mm.simulate_mm_system(system)
    plt.figure()
    plt.plot(data.x[:,0],'b.',label='True data')
    plt.plot(simDataEM.x[:,0],'r-',label='JMLS Sim. data')
    plt.grid()
    plt.legend()
    plt.title("Model fit")
    plt.show(block=False)


    return np.array([system, data, numSteps, avgExTime])

def jmls_fly_random(Nactions):
    Ts = 0.2
    sigma = 0.1

    #Data with correct mode assignments
    [data, noisy_data] = mm.jet_fly_random(Nactions, Ts, sigma)

    #Same data, but with random mode assignments
    dataCopy = copy.deepcopy(noisy_data)
    dataCopy.m = np.random.randint(3, size = data.m.size)

    system = mm.system_from_data(dataCopy)
    all_datasets = lmf.split_datasets(dataCopy)
    datasets_per_model = lmf.group_models(all_datasets)
    K = len(datasets_per_model)

    kernel_type='linear'
    mis_penalty = 1000000


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
        classifier_list[i] = classifier_list[i] = svm.SVC(kernel=kernel_type, C = mis_penalty )
        classifier_list[i].fit(svm_data_x, svm_data_m)

    for i in range(K):
        [A,B]=lmf.linear_model_fit(datasets_per_model[i])
        system.Am.append(A)
        print "Model "+str(i)+": A is \n"+str(A)

    for i in range(system.K):
        system.Am[i]+= np.random.rand(1)*0.001


    em.init_model(system,classifier_list,normalize_lik=True,scale_prob=True)
    max_steps = 20

    avg_likelihood = np.array([em.avg_log_likelihood(system)])
    mis_models = np.array([em.count_different(dataCopy.m[:-1],data.m[:-1])])

    numSteps = 0
    avgExTime = 0

    for i in range(max_steps):
        print "\nStep "+str(i)+""
        #E step 
        print "E step..."
        st = time.clock()
        em.compute_alphas(system)
        em.compute_betas(system)
        em.compute_gamma(system)
        em.compute_xi(system)

        #Estimates the most likely labels
        em.approx_viterbi(system)
        dataCopy.m = np.copy(system.m)

        
        print "M step..."
        em.max_initial_mode_probs(system)
        em.max_linear_models(system)
        em.max_guarded_trans_probs(system)
        avgExTime += time.clock()-st
    
        avg_likelihood = np.append(avg_likelihood,em.avg_log_likelihood(system))
        mis_models = np.append(mis_models,em.count_different(dataCopy.m[:-1],data.m[:-1]))

        print "Average log-likelihood: "+str(avg_likelihood[-1])
        print "Misclassified models: "+str(mis_models[-1])
        
        numSteps += 1
        if np.abs(avg_likelihood[-2]-avg_likelihood[-1])<5:
            print "\n\n********* Model converged *********\n\n"
            break

    avgExTime = avgExTime/numSteps

    simDataEM = mm.simulate_mm_system(system)
    plt.figure()
    plt.plot(data.x[:,0],'b.',label='True data')
    plt.plot(simDataEM.x[:,0],'r-',label='JMLS Sim. data')
    plt.grid()
    plt.legend()
    plt.title("Model fit")
    plt.show(block=False)


    return np.array([system, data, numSteps, avgExTime])



def generate_ci():
    jmls_data = np.zeros([30,1])
    for i in range(0, 30):
        jmls_data[i] = jmls()
    
    confidence=0.95
    n = len(jmls_data)
    m, se = np.mean(jmls_data), scipy.stats.sem(jmls_data)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

if __name__=="__main__":
    jmls()
