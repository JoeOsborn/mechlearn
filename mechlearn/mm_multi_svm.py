import mm_systems as mm
import numpy as np
import linear_model_fit as lmf
from sklearn import svm

class svm_data:
	def __init__(self)
		self.x = np.array([])
		self.y = np.array([])

class svm_classifier:
	def __init__(self):
		self.SVC = svm.SVC() 
		self.pMgivenRegion = np.array([]) #kxk matrix containing the probablity of transitioning to model j (the column number) given that the current estimated region is i (the row number)

def svm_fit(data, k):
	data_sets = mm.split_datasets(data)
	data_list = []
	classifier_list = []
	
	#init lists
	for i in range(0, k - 1):
		sData = svm_data()
		sData.model = i
		data_list.append(sData.model)
		
		sClassifier = svm_classifier()
		sClassifier.pMgivenRegion = np.zeros(k, k)
		classifier_list.append(sClassifier)
	
	#Populate the data_list
	for i in range(0, len(data_sets) - 2):
		model = data_sets{i].m[0]
		nextModel = data_sets[i + 1].m[0]
		model_list = ones(len(data_sets[i]))*model
		model_list[-1] = nextModel
		if len(data_list[model].x) == 0:
			data_list[model].x = np.copy(data_sets[i].x)
			data_list{model].y = np.copy(model_list)
		else:
			data_list[model].x.concatenate((data_list[model].x,data_sets[i].x), axis=0)
			data_list[model].y.concatenate((data_list[model].y,model_list), axis=0)

	#Add the last data_set (ignore the last element as it has no next model)
	model = data_sets[-1].m[0]
	model_list = ones(len(data_sets[-1] - 1) * model
	if len(data_list[model].x) == 0:
		data_list[model].x = np.copy(data_sets[-1].x)
		data_list{model].y = np.copy(model_list)
	else:
		data_list[model].x.concatenate((data_list[model].x,data_sets[-1].x[0:len(data_sets[-1])-2), axis=0)
	        data_list[model].y.concatenate((data_list[model].y,model_list), axis=0)

	#Estimate SVMs
	for i in range(0, k - 1):
		#Fit data
		classifier_list[i].SVC.fit(data_list[i].x, data_list[i].y]

		#Compute estimated labels of current dataset
		predictions = classifier_list[i].SVC.predict(data_list[i].x)
		
		#Count the number of elements in each region and where they transition to
		classifier_list[i].pMgivenRegion = np.zeros((k,k))
		counts = np.zeros(k)
		for j in range(0, len(data_list[i].x)):
			counts[preditions[j]] += 1
			classifier_list[i].pMgivenRegion[preditions[j],data_list[i].y[j]] += 1
		
		#Normalize the counts
		classifier_list[i].pMgivenRegion = numpy.transpose(numpy.transpose(classifier_list[i].pMgivenRegion)/counts)

	return classifier_list
