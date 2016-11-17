import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import mm_systems as mm
import linear_model_fit as lmf
import mm_svm

def regression_error_bbac(prob = -1.0):	
	Nsamples = 2000
        Ts = 0.3

	sqr_error = 0
	Ah = np.identity(2)
        Ah[0,1]=0.3*Ts #temperature increment

        #Colder
        Ac = np.identity(2)
        Ac[0,1]=-0.2*Ts #temperature decrement

	for i in range(100):
		#Whole data
        	data = mm.bang_bang_ac(Nsamples,Ts,prob)

	        #Datasets split at each transition
	        all_datasets = lmf.split_datasets(data)

        	#List of datasets grouped per model
	        datasets_per_model = lmf.group_models(all_datasets)

        	system = lmf.system_from_data(data)

	        for i in range(len(datasets_per_model)):
        	        [A,B]=lmf.linear_model_fit(datasets_per_model[i])
                	system.Am.append(A)
	                system.Bm.append(B) 
        	        #print "Model "+str(i)+": A is \n"+str(A)
		simData = mm.simulate_mm_system(system)
		sqr_error = np.sum(np.square(data.x - simData.x))
	
	return sqr_error/100

def svm_mse_bbac(Nsamples, prob = -1.0):
	sqr_error = 0        
        Ts = 0.3
	 

        for i in range(100):
		#Whole data
       		data = mm.bang_bang_ac(Nsamples,Ts,prob)

        	#Datasets split at each transition
        	all_datasets = lmf.split_datasets(data)

        	#List of datasets grouped per model
        	datasets_per_model = lmf.group_models(all_datasets)

                system = lmf.system_from_data(data)

                for i in range(len(datasets_per_model)):
                        [A,B]=lmf.linear_model_fit(datasets_per_model[i])
                        system.Am.append(A)
                        system.Bm.append(B)
                        #print "Model "+str(i)+": A is \n"+str(A)

		simData = mm.simulate_mm_system(system)
        	data_list = mm_svm.svm_split_data(simData, 2)
        	classifier_list = mm_svm.svm_fit(data_list, 2)

                error0 = np.square(-classifier_list[0][1].intercept_[0]/classifier_list[0][1].coef_[0][0] - 73)
                error1 = np.square(-classifier_list[1][0].intercept_[0]/classifier_list[1][0].coef_[0][0] - 70)

                sqr_error += error0 + error1
                
        return sqr_error/100
	
def cross_val(prob = -0.1):
	Nsamples = 2000
	Ts = 0.3
	training_data = mm.bang_bang_ac(Nsamples,Ts,prob)

	#Datasets split at each transition
        all_datasets = lmf.split_datasets(training_data)

        #List of datasets grouped per model
        datasets_per_model = lmf.group_models(all_datasets)

        system = lmf.system_from_data(training_data)

        for i in range(len(datasets_per_model)):
        	[A,B]=lmf.linear_model_fit(datasets_per_model[i])
                system.Am.append(A)
                system.Bm.append(B)
                #print "Model "+str(i)+": A is \n"+str(A)

        simData = mm.simulate_mm_system(system)
        data_list = mm_svm.svm_split_data(simData, 2)
        classifier_list = mm_svm.svm_fit(data_list, 2)
	
	training_counts = mm_svm.compute_counts(data_list[0][1].x, data_list[0][1].y, classifier_list[0][1])
	training_error = training_counts.count[1] + training_counts.count[3]
	training_counts = mm_svm.compute_counts(data_list[0][1].x, data_list[0][1].y, classifier_list[0][1])
        training_error += training_counts.count[1] + training_counts.count[3]
	training_error /= (Nsamples - 1)
	
	print training_error

	test_data = mm.bang_bang_ac(Nsamples,Ts,prob)
	data_list = mm_svm.svm_split_data(test_data, 2) 
	test_counts = mm_svm.compute_counts(data_list[0][1].x, data_list[0][1].y, classifier_list[0][1])
	test_error = test_counts.count[1] + test_counts.count[3]
	test_counts = mm_svm.compute_counts(data_list[0][1].x, data_list[0][1].y, classifier_list[0][1])
        test_error += test_counts.count[1] + test_counts.count[3]
	test_error /= (Nsamples - 1)
	print test_error
