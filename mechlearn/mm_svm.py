import mm_systems as mm
import numpy as np
import linear_model_fit as lmf
from sklearn import svm
import matplotlib.pyplot as pl
import pdb

class svm_data:
    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])

class svm_count_data:
    def __init__(self):
        self.count = np.zeros(4)
        self.norm_count = np.zeros(4)

def svm_split_data(data, K):
    
    #Splits the data sets according to mode transitions. It should be
    #noticed that each data set includes an additional state and input
    #vectors that pertain to the next mode. That's useful for the 
    #estimation of dynamical models, but should be ignored when computing
    #the transitions boundaries
    data_sets = lmf.split_datasets(data)
            
    #Data for the multiclass SVM with origin in each one of the modes
    svm_data_list = [ svm_data() for j in range(K)] 
            
    #Populate the data_list
    for i in range(len(data_sets)):
        
        if data_sets[i].u.size>0:
            newInputData = np.concatenate((data_sets[i].x[:-1],data_sets[i].u[:-1]),axis=1)         
        else:
            newInputData = np.copy(data_sets[i].x[:-1])
        
        currentModel = data_sets[i].m[0]        
        #The labels are given by the mode estimates, where the first
        #sample is ignored because states at time i only influence the
        #mode transition for i+1 (next time step), i.e., there is a one
        #sample shift between (x,u) and m
        newLabelData = data_sets[i].m[1:]               
                    
        #First time this entry has been filled
        if svm_data_list[currentModel].x.size == 0:                     
            svm_data_list[currentModel].x = newInputData
            svm_data_list[currentModel].y = newLabelData
        else:
            svm_data_list[currentModel].x = np.vstack((svm_data_list[currentModel].x,newInputData))
            svm_data_list[currentModel].y = np.concatenate((svm_data_list[currentModel].y,newLabelData))
    
    return svm_data_list

def svm_fit(svm_data_list, K, kernel_type='linear',svm_sample_weight=np.array([]), mis_penalty = 1000000, max_iter = 1000000):
    
    #Misclassification penalty
    #mis_penalty = 1000000
    #mis_penalty = 1000
    
    #SVM classifier for transitions with origin in mode i
    classifier_list =  [ [] for i in range(K)] 
    
    #Estimate SVMs
    for i in range(K):  
                #print "Starting" +str(i)
        if svm_data_list[i].x.shape[0] > 0:                 
            if svm_sample_weight.size == 0:
                classifier_list[i] = svm.SVC(kernel=kernel_type, C = mis_penalty, max_iter=max_iter)
                                #classifier_list[i] = svm.SVC(kernel=kernel_type, C = mis_penalty, class_weight={1: 1})
                #classifier_list[i] = svm.LinearSVC(C = mis_penalty, class_weight={1: 1})
            else:
                classifier_list[i] = svm.SVC(kernel=kernel_type, C = mis_penalty, sample_weight=svm_sample_weight, max_iter=max_iter)
                                #classifier_list[i] = svm.SVC(kernel=kernel_type, C = mis_penalty, sample_weight=svm_sample_weight)
                #classifier_list[i] = svm.LinearSVC(C = mis_penalty, sample_weight=svm_sample_weight)
            
            classifier_list[i].fit(svm_data_list[i].x, svm_data_list[i].y)  
        else:
            print "ERROR: empty dataset for SVM with origin in mode "+str(i)
    
    return classifier_list


def svm_plot(system):
    
    if system.u.size>0:
        svmTrainingData = np.hstack((system.x,system.u))
    else:
        svmTrainingData = system.x
    
    h = .02  # step size in the mesh
    
    #First two features of the training data
    Xplot = svmTrainingData[0:200,0]
    Yplot = svmTrainingData[0:200,1]

    x_min, x_max = Xplot.min() - 0.05, Xplot.max() + 0.05
    y_min, y_max =Yplot.min() - 0.05, Yplot.max() + 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
       # pdb.set_trace()    
    pl.figure()
    
    index = 1
    for mi_1 in range(system.K):        
        print 'Starting loop'                            
        pl.subplot(system.K,1,index)
        Z = system.guards[mi_1].predict(np.c_[xx.ravel(), yy.ravel()])
        
        if Z.size == 0:
            print "SVM["+str(mi_1)+"] gave empty predictions."
        
        Ztrain = system.guards[mi_1].predict(svmTrainingData)
        colors = ['w' for i in range(Ztrain.size)]

    #   pdb.set_trace()     

        for i in range(Ztrain.size):
            if Ztrain[i] == 1.0:
                colors[i] = 'k'
        #pdb.set_trace()                
        # Put the result into a color plot
        Z = Z.reshape(xx.shape) 
    
        #pdb.set_trace()
        pl.contour(xx,yy,Z,1,colors='k')
        pl.axis('on')

        # Plot also the training points`
        pl.scatter(Xplot, Yplot,c=colors)
        pl.title("mi_1="+str(mi_1))
        index+=1
        print index
        
#Returns np.array([correctly classified in active, incorrectly classified in active, correctly classified in inactive, incorrectly classified in inactive])
def compute_counts(X, y, clf):
    data = svm_count_data()
    yEst = clf.predict(X)
    for i in range(y.shape[0]):
        if yEst[i] == 1: #active
            if yEst[i] == y[i]: #correctly classified active
                data.count[0] += 1
            else: #incorrectly classified active
                data.count[1] += 1
        else: #inactive 
            if yEst[i] == y[i]: #correctly classified inactive
                data.count[2] += 1
            else: #incorrectly classified inactive
                data.count[3] += 1

    data.norm_count[0] = data.count[0]/np.sum(yEst)
    data.norm_count[1] = data.count[1]/np.sum(yEst)
    data.norm_count[2] = data.count[2]/(y.shape[0] - np.sum(yEst))
    data.norm_count[3] = data.count[3]/(y.shape[0] - np.sum(yEst)) 
    return data
            
    
