import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import mm_systems as mm
import linear_model_fit as lmf
import mm_svm
import em
import time
import mm_viterbi as vit
import copy
import pdb
import random

def test_simple_linsystem():
	Nsamples = 400
	data = mm.simple_linsystem(Nsamples)
	
	#Removes middle portion of the data
	#data.x = np.concatenate([data.x[0:50,:],data.x[100:,:]])
	#data.u = np.concatenate([data.u[0:50,:],data.u[100:,:]])
	
	[A,B] = lmf.linear_model_fit(data)
	print "A is \n"+str(A)
	print "B is \n"+str(B)

	system = mm.system_from_data(data)
	system.Am=[A]
	system.Bm=[B]
	
	simData = mm.simulate_mm_system(system)
	
	plt.plot(data.x[:,0],data.x[:,1],'b',label='True data')	
	plt.plot(simData.x[:,0],simData.x[:,1],'r',label='Sim data')	
	plt.show(block=False)
	
	return data


def test_jet_route_fly():
	Nsamples = 1000
	Ts = 0.2
	
	data = mm.jet_route_fly(Nsamples,Ts)	
	datasets = lmf.split_datasets(data)
	data_models = lmf.group_models(datasets)
			
	system = mm.system_from_data(data)
	
	for i in range(len(data_models)):
		[A,B]=lmf.linear_model_fit(data_models[i])
		system.Am.append(A)
		system.Bm.append(B)
		print "Model "+str(i)+": A is \n"+str(A)
	
	simData = mm.simulate_mm_system(system)
	
	plt.figure(1)
	plt.plot(data.x[:,0],data.x[:,2],'b',label='True data')			
	plt.plot(simData.x[:,0],simData.x[:,2],'r',label='Sim. data')	
	
	plt.show(block=False)	
	return data_models

def test_jet_fly_racetrack(Nloops):
	Ts = 0.2
	
	data = mm.jet_fly_racetrack(Nloops,Ts)
        Nsamples = data.num_x_samples
	#datasets = lmf.split_datasets(data)
	#data_models = lmf.group_models(datasets)
			
	#system = mm.system_from_data(data)
	
	#for i in range(len(data_models)):
	#	[A,B]=lmf.linear_model_fit(data_models[i])
	#	system.Am.append(A)
	#	system.Bm.append(B)
	#	print "Model "+str(i)+": A is \n"+str(A)
	
	#simData = mm.simulate_mm_system(system)
	
	plt.figure(1)
	plt.plot(data.x[:,0],data.x[:,2],'b',label='True data')			
	#plt.plot(simData.x[:,0],simData.x[:,2],'r',label='Sim. data')	
	
	plt.show(block=False)	
	return data

def test_jet_fly_random(Nactions):
	#Nactions = 10
	Ts = 0.2
	
	data = mm.jet_fly_random(Nactions,Ts)
        Nsamples = data.num_x_samples
	datasets = lmf.split_datasets(data)
	data_models = lmf.group_models(datasets)
	#pdb.set_trace()		

	#system = mm.system_from_data(data)
	
	#for i in range(len(data_models)):
	#	[A,B]=lmf.linear_model_fit(data_models[i])
	#	system.Am.append(A)
	#	system.Bm.append(B)
	#	print "Model "+str(i)+": A is \n"+str(A)
	
	#simData = mm.simulate_mm_system(system)
	
	plt.figure(1)
	plt.plot(data.x[:,0],data.x[:,2],'b',label='True data')			
	#plt.plot(simData.x[:,0],simData.x[:,2],'r',label='Sim. data')	
	
	plt.show(block=False)	
	return data


#def test_bang_bang_ac():	
##	Nsamples = 2000
#	Ts = 0.3
#	
	#Whole data
#	data = mm.bang_bang_ac(Nsamples,Ts,-1.0)
	
	#Datasets split at each transition
#	all_datasets = lmf.split_datasets(data)
	
	#List of datasets grouped per model
#	datasets_per_model = lmf.group_models(all_datasets)
			
#	system = mm.system_from_data(data)
	
#	for i in range(len(datasets_per_model)):		
#		[A,B]=lmf.linear_model_fit(datasets_per_model[i])
#		system.Am.append(A)
#		system.Bm.append(B)
#		print "Model "+str(i)+": A is \n"+str(A)
	
#	simData = mm.simulate_mm_system(system)
	
#	data_list = mm_svm.svm_split_data(simData, 2)
#	classifier_list = mm_svm.svm_fit(data_list, 2)
#	mm_svm.svm_plot(data_list, classifier_list, 2)

#	plt.figure(1)
#	plt.plot(data.x[:,0],'b',label='True data')			
#	plt.plot(simData.x[:,0],'r',label='Sim. data')	

 #       system.pm1 = np.array([1,0])
  #      alt_system = copy.deepcopy(system)
   #     vit.viterbi(system)

#	system.Wd = 3*np.var(mm.simulate_mm_system(system) - 
#
#	plt.show(block=False)	
	#return classifier_list
        return 1

def test_bang_bang_svm():
	Nsamples = 2000
	Ts = 0.3

	#Whole data
	data = mm.bang_bang_ac(Nsamples,Ts, -1.0)

	#Datasets split at each transition
	all_datasets = lmf.split_datasets(data)

	#List of datasets grouped per model
	datasets_per_model = lmf.group_models(all_datasets)
	
	system = mm.system_from_data(data)

	for i in range(len(datasets_per_model)):	
		[A,B]=lmf.linear_model_fit(datasets_per_model[i])
		system.Am.append(A)
		system.Bm.append(B)
		print "Model "+str(i)+": A is \n"+str(A)
	
	simData = mm.simulate_mm_system(system)
	data_list = mm_svm.svm_split_data(simData, 2)	
	classifier_list = mm_svm.svm_fit(data_list, 2)

	t = np.array(range(2000))
	t = t*Ts

	#Plot sim data
	plt.figure(1)
	plt.plot(t, data.x[:,0],'b',label='True data')	
	plt.plot(t, simData.x[:,0],'r',label='Sim. data')	

	#Plot transition 0 -> 1
	clf = classifier_list[0][1]
	w0 = clf.coef_[0]
	xx0 = np.array([0, 600])
	yy0 = np.array([-clf.intercept_[0] / w0[0], -clf.intercept_[0] / w0[0]])	
	h0 = plt.plot(xx0, yy0, 'b--', label='0->1')	

	#Plot transition 1 -> 0
	clf = classifier_list[1][0]
	w1 = clf.coef_[0]
	xx1 = np.array([0, 600])
	yy1 = np.array([-clf.intercept_[0] / w1[0], -clf.intercept_[0] / w1[0]])	
	h1 = plt.plot(xx1, yy1, 'r--', label='1->0')	

	plt.legend()
	plt.show()
	return classifier_list

def test_stoch_bang_bang_ac():	
	Nsamples = 2000
	Ts = 0.3

	#Whole data
	data = mm.bang_bang_ac(Nsamples,Ts,0.6)
	#Datasets split at each transition
	all_datasets = lmf.split_datasets(data)
	#List of datasets grouped per model
	datasets_per_model = lmf.group_models(all_datasets)
	system = mm.system_from_data(data)
	for i in range(len(datasets_per_model)):	
		[A,B]=lmf.linear_model_fit(datasets_per_model[i])
		system.Am.append(A)
		system.Bm.append(B)
		print "Model "+str(i)+": A is \n"+str(A)
	
	simData = mm.simulate_mm_system(system)
	data_list = mm_svm.svm_split_data(simData, 2)
	classifier_list = mm_svm.svm_fit(data_list, 2)
	mm_svm.svm_plot(data_list, classifier_list, 2)

	plt.figure(1)
	plt.plot(data.x[:,0],'b',label='True data')	
	plt.plot(simData.x[:,0],'r',label='Sim. data')	
	plt.show(block=False)	
	return classifier_list


def test_rc_circuit():	
	#~ Nsamples = 5000
	#~ Ts = 1.0
	#~ Rl=10000.0; Rc=100; C=0.3
	#~ X0=3.8; Vin=10.0; Vmin=3.0; Vmax=4.0
	
	Nsamples = 3000
	Ts = 1.0
	Rl=2000.0; Rc=400; C=0.3
	X0=3.8; Vin=10.0; Vmin=3.0; Vmax=4.0
	
	#Whole data
	data = mm.rc_circuit(Nsamples,Ts,Rl,Rc,C,X0,Vin,Vmin,Vmax,prob=-1.0)
	#Datasets split at each transition
	all_datasets = lmf.split_datasets(data)
	#List of datasets grouped per model
	datasets_per_model = lmf.group_models(all_datasets)
	system = mm.system_from_data(data)
	for i in range(len(datasets_per_model)):	
		[A,B]=lmf.linear_model_fit(datasets_per_model[i])
		system.Am.append(A)
		system.Bm.append(B)
		#print "Model "+str(i)+": A is \n"+str(A)
		#print "Model "+str(i)+": B is \n"+str(B)
	
	#simData = mm.simulate_mm_system(system)
	#data_list = mm_svm.svm_split_data(simData, 2)
	#classifier_list = mm_svm.svm_fit(data_list, 2)
    
	#plt.figure(1)
	#plt.plot(data.x[:,0],'b.',label='True data')	
	#plt.plot(simData.x[:,0],'r',label='Sim. data')	
	#plt.legend()
	#plt.show(block=False)	
	#return classifier_list
        #em.max_guarded_trans_probs(system)
        #system.pm1 = np.array([1,0])
        #est_system = vit.log_viterbi(system)

        return data

def test_rc_circuit_series(Ncircuits, Nsamples = 5000):	
	#~ Nsamples = 5000
	#~ Ts = 1.0
	#~ Rl=10000.0; Rc=100; C=0.3
	#~ X0=3.8; Vin=10.0; Vmin=3.0; Vmax=4.0
	
	Ts = 1.0
	Rl=2000.0; Rc=400; C=0.3
	X0=3.8; Vin=10.0; Vmin=3.0; Vmax=4.0
	
	#Whole data
	data = mm.rc_circuit_series(Ncircuits,Nsamples,Ts,Rl,Rc,C,X0,Vin,Vmin,Vmax,prob=-1.0)
	#Datasets split at each transition
	#all_datasets = lmf.split_datasets(data)
	#List of datasets grouped per model
	#pdb.set_trace()
        #datasets_per_model = lmf.group_models(all_datasets)
	#system = mm.system_from_data(data)
	#for i in range(len(datasets_per_model)):	
	#	[A,B]=lmf.linear_model_fit(datasets_per_model[i])
        #	system.Am.append(A)
	#	system.Bm.append(B)
		#print "Model "+str(i)+": A is \n"+str(A)
		#print "Model "+str(i)+": B is \n"+str(B)
	
	#simData = mm.simulate_mm_system(system)
	#data_list = mm_svm.svm_split_data(simData, 2)
	#classifier_list = mm_svm.svm_fit(data_list, 2)
    
        for i in range(0, Ncircuits):
	    plt.figure()
	    plt.plot(data.x[:,i],'b.',label='True data')	
	#plt.plot(simData.x[:,0],'r',label='Sim. data')	
	plt.figure()
        plt.plot(data.m[:],'b.')
        #plt.legend()
	plt.show(block=False)	
	#return classifier_list
        
        return data


def test_em(Nsamples, p=-1):	
	
	# Simulated data from the dynamical system, where we assume that
	# both x and u are observable.
	
	#~ Nsamples = 500
	#~ Ts = 0.3	
	#~ data = mm.bang_bang_ac(Nsamples,Ts,-1.0)
		
	#~ Nsamples = 500
	#~ Ts = 1.0
	#~ Rl=2000.0; Rc=400; C=0.3
	#~ X0=3.8; Vin=10.0; Vmin=3.0; Vmax=4.0		
	#~ data = mm.rc_circuit(Nsamples,Ts,Rl,Rc,C,X0,Vin,Vmin,Vmax,prob=0.5)

	Ts = 1.0
	Rl=2000.0; Rc=400; C=0.3
	X0=3.5; Vin=10.0; Vmin=3.0; Vmax=4.0; sigma = 0.05		
	
	#Data with correct mode assignments
	[data, noisy_data] = mm.rc_circuit(Nsamples,Ts,Rl,Rc,C,X0,Vin,Vmin,Vmax,sigma,prob=-1.0)
		
	#Same data, but with random mode assignments
	dataCopy = copy.deepcopy(noisy_data)	
	dataCopy.m = np.random.binomial(1,0.5,noisy_data.m.size)
	
	#Initial number of mislabeled points
	#initial_mislab = em.count_different(dataCopy.m[:-1],data.m[:-1])
	
	#Stores the experimental data
	system = mm.system_from_data(dataCopy)
		
	#~ # These lines are cheating, since I'm splitting the datasets
	#~ # according to the true mode distributions. 
	#~ ######
	#~ #Datasets split at each transition
	#~ all_datasets = lmf.split_datasets(data)	
	#~ #List of datasets grouped per model
	#~ datasets_per_model = lmf.group_models(all_datasets)
	#~ K = len(datasets_per_model) #True number of modes in the dataset	
	#~ ######		
		
	#Datasets split at each transition using random assignments
	#(it doesn't get worse than that)
	all_datasets = lmf.split_datasets(dataCopy)	
	#List of datasets grouped per model
	datasets_per_model = lmf.group_models(all_datasets)
	K = len(datasets_per_model) #True number of modes in the dataset	
	######		
	
	print "***** Initial model estimates *****"
	
	#Initial estimate of the parameters using completely random data
	for i in range(K):		
		[A,B]=lmf.linear_model_fit(datasets_per_model[i])
		system.Am.append(A)		
		print "Model "+str(i)+": A is \n"+str(A)		
		if B != None:
			system.Bm.append(B)		
			print "Model "+str(i)+": B is \n"+str(B)+"\n"
	
	#~ #Estimates the activation regions using perfectly-separated data
	#~ #simData = mm.simulate_mm_system(system)	
	#~ data_list = mm_svm.svm_split_data(data, K)
	#~ classifier_list = mm_svm.svm_fit(data_list, K)
	
	print "***** Done *****\n"
	
	print "***** Initial guard conditions *****"
	
	#Estimates the activation regions using heavily corrupted data	
	data_list = mm_svm.svm_split_data(dataCopy, K)
	
        #print "Split Data"

        classifier_list = mm_svm.svm_fit(data_list, K)
		
	print "***** Done *****\n"
			
	#~ #Disturbs the models a little bit	
	#~ for i in range(system.K):
		#~ system.Am[i]+= np.random.rand(1)*0.005
		#~ system.Bm[i]+= np.random.rand(1)*0.005
	    
	#Disturbs the models a little bit	
	for i in range(system.K):
		system.Am[i]+= np.random.rand(1)*0.005
		system.Bm[i]+= np.random.rand(1)*0.005
	
	
	simDataDisturbed = mm.simulate_mm_system(system)
	
	em.init_model(system,classifier_list,normalize_lik=True,scale_prob=True)

	max_steps = 200
		
	#Initial model likelihood
	avg_likelihood = np.array([em.avg_log_likelihood(system)])
	#Initial number of misclassified models
	mis_models = np.array([em.count_different(dataCopy.m[:-1],data.m[:-1])])
	
	#Plots initial guard conditions
	#mm_svm.svm_plot(system)
	numSteps = 0
        avgExTime = 0
        conv_count=0
	for i in range(max_steps):
 
		print "\nStep "+str(i)+""
		#E step	
		print "E step..."
		st = time.clock()
		em.compute_alphas(system)
		em.compute_betas(system)
		em.compute_gamma(system)
		em.compute_xi(system)

		
		print "SVM step..."
		
                #Estimates the most likely labels
		#~ for mi_1 in range(K):
			#~ training_data = em.em_svm_training(mi_1,system)
			#~ system.guards[mi_1] = em.em_svm_fit(training_data, K, kernel_type='linear')
		
		em.approx_viterbi(system)
		dataCopy.m = np.copy(system.m)		
	        		
		#Estimates the activation regions using estimated data
		data_list = mm_svm.svm_split_data(dataCopy, K)
		
                classifier_list = mm_svm.svm_fit(data_list, K)
		system.guards = classifier_list
				
		#M step
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
                
	
	simDataEM = mm.simulate_mm_system(system)
		
	#Initial number of mislabeled points
	#final_mislab = em.count_different(dataCopy.m[:-1],data.m[:-1])
	
	avgExTime = avgExTime/numSteps
	#Plotting	
	#plt.figure()
	#plt.plot(np.arange(1,avg_likelihood.size),avg_likelihood[1:],'bo-')
	#plt.grid()
	#plt.xlabel("Step number")
	#plt.ylabel("Avg. log-likelihood")
	#plt.title("Evolution of average log-likelihood")
	
	#plt.figure()
	#if mis_models[-1] > (mis_models.size/2.0):
	#	plt.plot(np.arange(1,mis_models.size+1),system.n-mis_models,'bo-')
	#else:
        #	plt.plot(np.arange(1,mis_models.size+1),mis_models,'bo-')
	#plt.grid()
	#plt.xlabel("Step number")
	#plt.ylabel("Number of mislabeled data points")
	#plt.title("Evolution of mislabeled data points")
	
	#mm_svm.svm_plot(system)
		
	plt.figure()
	plt.plot(noisy_data.x[:,0],'b.',label='True data')	
	plt.plot(simDataEM.x[:,0],'r-',label='EM Sim. data')
	#plt.plot(simDataDisturbed.x[:,0],'m-',label='Dist. Sim. data')	
	plt.grid()
	plt.legend()
	plt.title("Model fit")
	plt.show(block=False)	
	
	#alt_system = copy.deepcopy(system)
        #vit.log_viterbi(system)

	#return np.sum(np.logical_not(np.equal(alt_system.m, system.m)))
        #return np.array([alt_system.m, system.m])
        #return system
        return np.array([system, data, noisy_data, numSteps, avgExTime])

def test_flight_em(Nactions):	
	
	# Simulated data from the dynamical system, where we assume that
	# both x and u are observable.
	
	Ts = 0.2
	
	#Data with correct mode assignments
	data = mm.jet_fly_random(Nactions, Ts)
		
	#Same data, but with random mode assignments
	dataCopy = copy.deepcopy(data)	
	dataCopy.m = np.random.randint(3, size = data.m.size)
        

	#Initial number of mislabeled points
	#initial_mislab = em.count_different(dataCopy.m[:-1],data.m[:-1])
	
	#Stores the experimental data
	system = mm.system_from_data(dataCopy)
				
	#Datasets split at each transition using random assignments
	#(it doesn't get worse than that)
	all_datasets = lmf.split_datasets(dataCopy)	
	#List of datasets grouped per model
	datasets_per_model = lmf.group_models(all_datasets)
	K = len(datasets_per_model) #True number of modes in the dataset	
	######		

        #print K

	print "***** Initial model estimates *****"
	
	#Initial estimate of the parameters using completely random data
	for i in range(K):		
		[A,B]=lmf.linear_model_fit(datasets_per_model[i])
		system.Am.append(A)		
		print "Model "+str(i)+": A is \n"+str(A)		
		if B != None:
			system.Bm.append(B)		
			print "Model "+str(i)+": B is \n"+str(B)+"\n"	
	print "***** Done *****\n"
	
	print "***** Initial guard conditions *****"
	
	#Estimates the activation regions using heavily corrupted data	
	data_list = mm_svm.svm_split_data(dataCopy, K)
        #pdb.set_trace()	
        #print "Split Data"

        classifier_list = mm_svm.svm_fit(data_list, K)
		
	print "***** Done *****\n"
			
	#Disturbs the models a little bit	
	#pdb.set_trace()
        for i in range(K):
		system.Am[i]+= np.random.rand(1)*0.001
    #		system.Bm[i]+= np.random.rand(1)*0.001
	
	
	simDataDisturbed = mm.simulate_mm_system(system)
	
	#Initializes EM with the true model (things should behave!)
	em.init_model(system,classifier_list,normalize_lik=True,scale_prob=True)

	max_steps = 200
		
	#Initial model likelihood
	avg_likelihood = np.array([em.avg_log_likelihood(system)])
	#Initial number of misclassified models
	mis_models = np.array([em.count_different(dataCopy.m[:-1],data.m[:-1])])

        numSteps = 0
        conv_count = 0
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
		print "Elapsed time [s]: "+str(time.clock()-st)+""
		
		print "SVM step..."
		st = time.clock()
		
		vit.log_viterbi(system)
		dataCopy.m = np.copy(system.m)		
	        #print "Copied"			
		#Estimates the activation regions using estimated data
		data_list = mm_svm.svm_split_data(dataCopy, K)
		#print "Split"
                #pdb.set_trace()
                classifier_list = mm_svm.svm_fit(data_list, K)
        	system.guards = classifier_list
		print "Elapsed time [s]: "+str(time.clock()-st)+""
		
		#M step
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
		
                numSteps += 1
        	if np.abs(avg_likelihood[-2]-avg_likelihood[-1])<5:
                    conv_count+=1
                else:
                    conv_count = 0


                if conv_count >= 15:
                    print "\n\n********* Model converged *********\n\n"
                    break
                
	

        simDataEM = mm.simulate_mm_system(system)
	
#	initial_rmse = em.rmse(data.x,simDataDisturbed.x)
#	final_rmse = em.rmse(data.x,simDataEM.x)
#	
#	#Initial number of mislabeled points
#	#final_mislab = em.count_different(dataCopy.m[:-1],data.m[:-1])
#
#	
#	print "\n\n***** Initial RMSE= "+str(initial_rmse)
#	print "***** Final RMSE= "+str(final_rmse)+"\n\n"
#	
#	print "***** Initial mislabeled points= "+str(mis_models[0])
#	print "***** Final mislabeled points= "+str(mis_models[-1])+"\n\n"
#	
#	
#	#Plotting	
#	plt.figure()
#	plt.plot(np.arange(1,avg_likelihood.size),avg_likelihood[1:],'bo-')
#	plt.grid()
#	plt.xlabel("Step number")
#	plt.ylabel("Avg. log-likelihood")
#	plt.title("Evolution of average log-likelihood")
#	
#	plt.figure()
#	if mis_models[-1] > (mis_models.size/2.0):
#		plt.plot(np.arange(1,mis_models.size+1),system.n-mis_models,'bo-')
#	else:
#		plt.plot(np.arange(1,mis_models.size+1),mis_models,'bo-')
#	plt.grid()
#	plt.xlabel("Step number")
#	plt.ylabel("Number of mislabeled data points")
#	plt.title("Evolution of mislabeled data points")
#	
#	#mm_svm.svm_plot(system)

        plt.figure()
	plt.plot(data.x[:,0],data.x[:,2],'b.',label='True data')	
	plt.plot(simDataEM.x[:,0],simDataEM.x[:,2],'r-',label='EM Sim. data')
	#plt.plot(simDataDisturbed.x[:,0],simDataDisturved.x[:,2],'m-',label='Dist. Sim. data')	
	plt.grid()
	plt.legend()
	plt.title("Model fit")
	plt.show(block=False)	
	
	#alt_system = copy.deepcopy(system)
        #vit.log_viterbi(system)

	#return np.sum(np.logical_not(np.equal(alt_system.m, system.m)))
        #return np.array([alt_system.m, system.m])
        return np.array([system, data, numSteps])





def test_racetrack_em(Nloops):	
	
	# Simulated data from the dynamical system, where we assume that
	# both x and u are observable.
	
	Ts = 0.2
	
	#Data with correct mode assignments
	data = mm.jet_fly_racetrack(Nloops, Ts)
		
	#Same data, but with random mode assignments
	dataCopy = copy.deepcopy(data)	
	dataCopy.m = np.random.randint(3, size = data.m.size)
        

	#Initial number of mislabeled points
	#initial_mislab = em.count_different(dataCopy.m[:-1],data.m[:-1])
	
	#Stores the experimental data
	system = mm.system_from_data(dataCopy)
				
	#Datasets split at each transition using random assignments
	#(it doesn't get worse than that)
	all_datasets = lmf.split_datasets(dataCopy)	
	#List of datasets grouped per model
	datasets_per_model = lmf.group_models(all_datasets)
	K = len(datasets_per_model) #True number of modes in the dataset	
	######		

        #print K

	print "***** Initial model estimates *****"
	
	#Initial estimate of the parameters using completely random data
	for i in range(K):		
		[A,B]=lmf.linear_model_fit(datasets_per_model[i])
		system.Am.append(A)		
		print "Model "+str(i)+": A is \n"+str(A)		
		if B != None:
			system.Bm.append(B)		
			print "Model "+str(i)+": B is \n"+str(B)+"\n"	
	print "***** Done *****\n"
	
	print "***** Initial guard conditions *****"
	
	#Estimates the activation regions using heavily corrupted data	
	data_list = mm_svm.svm_split_data(dataCopy, K)
        #pdb.set_trace()	
        #print "Split Data"

        classifier_list = mm_svm.svm_fit(data_list, K)
		
	print "***** Done *****\n"
			
	#Disturbs the models a little bit	
	#pdb.set_trace()
        for i in range(K):
		system.Am[i]+= np.random.rand(1)*0.001
    #		system.Bm[i]+= np.random.rand(1)*0.001
	
	
	simDataDisturbed = mm.simulate_mm_system(system)
	
	#Initializes EM with the true model (things should behave!)
	em.init_model(system,classifier_list,normalize_lik=True,scale_prob=True)

	max_steps = 20
		
	#Initial model likelihood
	avg_likelihood = np.array([em.avg_log_likelihood(system)])
	#Initial number of misclassified models
	mis_models = np.array([em.count_different(dataCopy.m[:-1],data.m[:-1])])

        numSteps = 0

	for i in range(max_steps):
 
		print "\nStep "+str(i)+""
		#E step	
		print "E step..."
		st = time.clock()
		em.compute_alphas(system)
		em.compute_betas(system)
		em.compute_gamma(system)
        	em.compute_xi(system)
		print "Elapsed time [s]: "+str(time.clock()-st)+""
		
		print "SVM step..."
		st = time.clock()
		
		em.approx_viterbi(system)
		dataCopy.m = np.copy(system.m)		
	        #print "Copied"			
		#Estimates the activation regions using estimated data
		data_list = mm_svm.svm_split_data(dataCopy, K)
		#print "Split"
                #pdb.set_trace()
                
                classifier_list = mm_svm.svm_fit(data_list, K)
        	system.guards = classifier_list
		print "Elapsed time [s]: "+str(time.clock()-st)+""
		
		#M step
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
		
                numSteps += 1

		if np.abs(avg_likelihood[-2]-avg_likelihood[-1])<5:
			print "\n\n********* Model converged *********\n\n"
			break		
	
        simDataEM = mm.simulate_mm_system(system)
	
#	initial_rmse = em.rmse(data.x,simDataDisturbed.x)
#	final_rmse = em.rmse(data.x,simDataEM.x)
#	
#	#Initial number of mislabeled points
#	#final_mislab = em.count_different(dataCopy.m[:-1],data.m[:-1])
#
#	
#	print "\n\n***** Initial RMSE= "+str(initial_rmse)
#	print "***** Final RMSE= "+str(final_rmse)+"\n\n"
#	
#	print "***** Initial mislabeled points= "+str(mis_models[0])
#	print "***** Final mislabeled points= "+str(mis_models[-1])+"\n\n"
#	
#	
#	#Plotting	
#	plt.figure()
#	plt.plot(np.arange(1,avg_likelihood.size),avg_likelihood[1:],'bo-')
#	plt.grid()
#	plt.xlabel("Step number")
#	plt.ylabel("Avg. log-likelihood")
#	plt.title("Evolution of average log-likelihood")
#	
#	plt.figure()
#	if mis_models[-1] > (mis_models.size/2.0):
#		plt.plot(np.arange(1,mis_models.size+1),system.n-mis_models,'bo-')
#	else:
#		plt.plot(np.arange(1,mis_models.size+1),mis_models,'bo-')
#	plt.grid()
#	plt.xlabel("Step number")
#	plt.ylabel("Number of mislabeled data points")
#	plt.title("Evolution of mislabeled data points")
#	
#	#mm_svm.svm_plot(system)

        plt.figure()
	plt.plot(data.x[:,0],data.x[:,2],'b.',label='True data')	
	plt.plot(simDataEM.x[:,0],simDataEM.x[:,2],'r-',label='EM Sim. data')
	#plt.plot(simDataDisturbed.x[:,0],simDataDisturved.x[:,2],'m-',label='Dist. Sim. data')	
	plt.grid()
	plt.legend()
	plt.title("Model fit")
	plt.show(block=False)	
	
	#alt_system = copy.deepcopy(system)
        #vit.log_viterbi(system)

	#return np.sum(np.logical_not(np.equal(alt_system.m, system.m)))
        #return np.array([alt_system.m, system.m])
        return np.array([system, data, numSteps])

def test_series_em(Nloops):	
	
	# Simulated data from the dynamical system, where we assume that
	# both x and u are observable.
	
	Ts = 0.2
	
	#Data with correct mode assignments
	data = mm.jet_fly_racetrack(Nloops, Ts)
		
	#Same data, but with random mode assignments
	dataCopy = copy.deepcopy(data)	
	dataCopy.m = np.random.randint(3, size = data.m.size)
        

	#Initial number of mislabeled points
	#initial_mislab = em.count_different(dataCopy.m[:-1],data.m[:-1])
	
	#Stores the experimental data
	system = mm.system_from_data(dataCopy)
				
	#Datasets split at each transition using random assignments
	#(it doesn't get worse than that)
	all_datasets = lmf.split_datasets(dataCopy)	
	#List of datasets grouped per model
	datasets_per_model = lmf.group_models(all_datasets)
	K = len(datasets_per_model) #True number of modes in the dataset	
	######		

        #print K

	print "***** Initial model estimates *****"
	
	#Initial estimate of the parameters using completely random data
	for i in range(K):		
		[A,B]=lmf.linear_model_fit(datasets_per_model[i])
		system.Am.append(A)		
		print "Model "+str(i)+": A is \n"+str(A)		
		if B != None:
			system.Bm.append(B)		
			print "Model "+str(i)+": B is \n"+str(B)+"\n"	
	print "***** Done *****\n"
	
	print "***** Initial guard conditions *****"
	
	#Estimates the activation regions using heavily corrupted data	
	data_list = mm_svm.svm_split_data(dataCopy, K)
        #pdb.set_trace()	
        #print "Split Data"

        classifier_list = mm_svm.svm_fit(data_list, K)
		
	print "***** Done *****\n"
			
	#Disturbs the models a little bit	
	#pdb.set_trace()
        for i in range(K):
		system.Am[i]+= np.random.rand(1)*0.001
    #		system.Bm[i]+= np.random.rand(1)*0.001
	
	
	simDataDisturbed = mm.simulate_mm_system(system)
	
	#Initializes EM with the true model (things should behave!)
	em.init_model(system,classifier_list,normalize_lik=True,scale_prob=True)

	max_steps = 20
		
	#Initial model likelihood
	avg_likelihood = np.array([em.avg_log_likelihood(system)])
	#Initial number of misclassified models
	mis_models = np.array([em.count_different(dataCopy.m[:-1],data.m[:-1])])
	
        numSteps = 0

	for i in range(max_steps):
 
		print "\nStep "+str(i)+""
		#E step	
		print "E step..."
		st = time.clock()
		em.compute_alphas(system)
		em.compute_betas(system)
		em.compute_gamma(system)
        	em.compute_xi(system)
		print "Elapsed time [s]: "+str(time.clock()-st)+""
		
		print "SVM step..."
		st = time.clock()
		
		em.approx_viterbi(system)
		dataCopy.m = np.copy(system.m)		
	        #print "Copied"			
		#Estimates the activation regions using estimated data
		data_list = mm_svm.svm_split_data(dataCopy, K)
		#print "Split"
                #pdb.set_trace()
                
                classifier_list = mm_svm.svm_fit(data_list, K)
        	system.guards = classifier_list
		print "Elapsed time [s]: "+str(time.clock()-st)+""
		
		#M step
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
		
                numSteps += 1

		if np.abs(avg_likelihood[-2]-avg_likelihood[-1])<5:
			print "\n\n********* Model converged *********\n\n"
			break		
	
        simDataEM = mm.simulate_mm_system(system)
	
#	initial_rmse = em.rmse(data.x,simDataDisturbed.x)
#	final_rmse = em.rmse(data.x,simDataEM.x)
#	
#	#Initial number of mislabeled points
#	#final_mislab = em.count_different(dataCopy.m[:-1],data.m[:-1])
#
#	
#	print "\n\n***** Initial RMSE= "+str(initial_rmse)
#	print "***** Final RMSE= "+str(final_rmse)+"\n\n"
#	
#	print "***** Initial mislabeled points= "+str(mis_models[0])
#	print "***** Final mislabeled points= "+str(mis_models[-1])+"\n\n"
#	
#	
#	#Plotting	
#	plt.figure()
#	plt.plot(np.arange(1,avg_likelihood.size),avg_likelihood[1:],'bo-')
#	plt.grid()
#	plt.xlabel("Step number")
#	plt.ylabel("Avg. log-likelihood")
#	plt.title("Evolution of average log-likelihood")
#	
#	plt.figure()
#	if mis_models[-1] > (mis_models.size/2.0):
#		plt.plot(np.arange(1,mis_models.size+1),system.n-mis_models,'bo-')
#	else:
#		plt.plot(np.arange(1,mis_models.size+1),mis_models,'bo-')
#	plt.grid()
#	plt.xlabel("Step number")
#	plt.ylabel("Number of mislabeled data points")
#	plt.title("Evolution of mislabeled data points")
#	
#	#mm_svm.svm_plot(system)

        plt.figure()
	plt.plot(data.x[:,0],data.x[:,2],'b.',label='True data')	
	plt.plot(simDataEM.x[:,0],simDataEM.x[:,2],'r-',label='EM Sim. data')
	#plt.plot(simDataDisturbed.x[:,0],simDataDisturved.x[:,2],'m-',label='Dist. Sim. data')	
	plt.grid()
	plt.legend()
	plt.title("Model fit")
	plt.show(block=False)	
	
	#alt_system = copy.deepcopy(system)
        #vit.log_viterbi(system)

	#return np.sum(np.logical_not(np.equal(alt_system.m, system.m)))
        #return np.array([alt_system.m, system.m])
        return np.array([system, data, numSteps])


def test_viterbi(Nsamples, Nreps, p=-1):
    sum = 0.0

    for i in range(0, Nreps):
        a = test_em(Nsamples, p)
        sum += np.sum(np.logical_not(np.equal(a[0], a[1])))
    return ((sum/Nreps)/Nsamples)*100.0


if __name__=="__main__":
	test_em(1000)


