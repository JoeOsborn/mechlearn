import EKF_filtering
import IMM_filtering
import numpy as np
import pdb
import pickle
import numpy.linalg as la
import matplotlib.pyplot as plt
import jmls_data
import test_systems as ts
import itertools
import copy
import mm_svm
import time

class cPHA_Filter:
    def __init__(self, system, Rd, Wd, data, true_data, is_jmls):
        self.data = data
        self.system = system
        self.true_data = true_data
        self.is_jmls = is_jmls
        self.Rd = Rd
        self.Wd = Wd

    def filter_system(self):
        #Initialize filter_list
        filter_list = np.array([])

        for i in range(self.system.K):
            m = int(i)
            pf = lambda x_prev, u, m = i: self.propag_function(m, x_prev, u)
            pl = lambda x_prev, u, m = i: self.propag_lin(m, x_prev, u)
            mf = lambda x_prev: self.measure_function(x_prev)
            ml = lambda x_prev: self.measure_lin(x_prev)
            
            new_filter = EKF_filtering.EKF(pf, pl, mf, ml)
            
            new_filter.init(self.data.x[0], np.identity(len(self.system.x[0,:]))*10)
            new_filter.set_Rd(self.Rd[i])
            new_filter.set_Wd(self.Wd[i])
            print 'R' + str(i) +': ' + str(self.Rd[i])
            print 'W' + str(i) +': ' + str(self.Wd[i])
            filter_list = np.append(filter_list, new_filter)

        #Initialize data lists
        filtered_x = np.zeros(self.data.x.shape)
        filtered_m = np.zeros(self.data.m.shape)
    
        #Initialize IMM Filter        
        fTPM = lambda u, x: self.get_TPM(u,x)
        imm_filter = IMM_filtering.IMM_Filter(self.system, self.data, filter_list, fTPM)
        filtered_x[0] = self.data.x[0]

        st = time.clock()
        #Run IMM Filter
        for i in range(1, len(self.data.m)):
            #print 'Starting step: ' + str(i)
            filtered_x[i], filtered_m[i] = imm_filter.get_next_estimate()
        print "Avg. Filter Time: " + str((time.clock()-st)/float(len(self.data.m)))       
 
        #Calculate RMSE
        rmse = la.norm(filtered_x - self.true_data.x)/np.sqrt(len(self.data.m))

        #Calculate Misclassification rate
        min_error,min_error_filtered_modes = min_error_mode_mapping(self.true_data.m,filtered_m,self.system.K)

#   min_error = len(self.true_data.m)+1
#   for mode_map in itertools.permutations(range(self.system.K)):
#       err_count = np.sum([1 for rm,m in zip([mode_map[int(fm)] for fm in filtered_m],self.true_data.m) if rm!=m])
        #print err_count
#       min_error = np.min([err_count,min_error])
    
        print rmse
        print min_error
        return filtered_x, filtered_m, min_error_filtered_modes

    def propag_function(self, m, x_prev, u):
        x = None
        if u != None:
            x = np.dot(self.system.Am[m],x_prev) + np.dot(self.system.Bm[m],u)
        else:
            x = np.dot(self.system.Am[m],x_prev)

        return x

    def propag_lin(self, m, x_prev, u):
        jacobian = np.copy(self.system.Am[m])   
    
        #Add the effects of Bu
        #if u != None:
        #    for i in range(len(x_prev)):
        #        jacobian[:, i] += np.dot(self.system.Bm[m],u)

        #Add the effects of the rest of the x elements
        #for i in range(len(x_prev)): 
        #    x_temp = np.copy(x_prev)
        #    x_temp[i] = 0
        #    jacobian[:, i] += np.dot(self.system.Am[m],x_temp)
    
        return jacobian

    def measure_function(self, x_prev):
        return x_prev

    def measure_lin(self, x_prev):
        return np.identity(len(x_prev))


    def get_TPM(self, u, x):
        TPM = np.identity(self.system.K)
        if self.is_jmls:
            for m in  range(self.system.K):
                TPM[m] = self.system.ptrans[m, 0, :]
        else:
            curr_data = None
            if u != None:
                curr_data = np.concatenate((x,u), axis = 0) 
            else:
                curr_data = x

             #Update TPM
            for m in  range(self.system.K):
                TPM[m,:] = self.system.ptrans[m,self.system.guards[m].predict(curr_data),:]
        #TPM = np.ones(TPM.shape)*(1.0/self.system.K) #This is a dirty, dirty hack            
        
        #IMM Expects column->row  
        TPM = TPM.transpose()
        return TPM

def min_error_mode_mapping(true_m,filtered_m,num_modes):
    min_error = len(true_m)+1
    min_remap_m = filtered_m
    #pdb.set_trace()
    for mode_map in itertools.permutations(range(num_modes)):
        remap_m = [mode_map[int(fm)] for fm in filtered_m]
        err_count = np.sum([1 for rm,m in zip(remap_m,true_m) if rm!=m])
        
        if err_count < min_error:
            min_error = err_count
            min_remap_m = remap_m
 
    return min_error,min_remap_m

def set_W_R(system, data, sigmaR, sigmaW = None):
    Rd = []
    for i in range(system.K):
        Rd.append(np.diag([e**2 for e in sigmaR]))   
    
    Wd = []
    if sigmaW == None:
        system.calculate_W(data)
        for i in range(system.K):
            Wd.append(np.copy(system.W))
    else:
        for i in range(system.K):
            Wd.append(np.diag([e**2 for e in sigmaW]))

    return Rd, Wd

def learn():
    [system, data, noisy_data, junk1, junk2] = ts.rc_circuit_em(1000, 0.02, 0.01/3.0, False)
    print junk2
    print np.sum(np.abs(system.m - data.m))
    with open('em_circuit.pkl', 'w') as output:
        pickle.dump(system,output)
        pickle.dump(data,output)
        pickle.dump(noisy_data, output)
    return system, data, noisy_data
 
def test(sigmaR = [0.05], sigmaW = [0.05]):
    #[system, data, junk1, junk2] = testing.test_em(1000)
   
    with open('em_circuit.pkl','r') as infile:
        system = pickle.load(infile)
        data = pickle.load(infile)
        noisy_data = pickle.load(infile)

    sensor_data = copy.deepcopy(noisy_data)
    Rd, Wd = set_W_R(system, data, sigmaR = sigmaR, sigmaW = sigmaW)
    cFilter = cPHA_Filter(system = system, Rd = Rd, Wd = Wd, data = sensor_data, true_data = data, is_jmls = False)
    filtered_x, filtered_m, min_error_filtered_m = cFilter.filter_system()

    misclass = np.abs(filtered_m - data.m)
    misclassInv = np.ones(misclass.shape) - misclass
    mis_ind = misclass if np.sum(misclass) < np.sum(misclassInv) else misclassInv
    print "Misclass: " + str(np.sum(mis_ind))   
 
    plt.figure()
#    plt.subplot(211)
    plt.plot(noisy_data.x[:,0],'b.',label='Sensor data')
    plt.plot(filtered_x[:,0],'r.',label='Filtered data')
    plt.plot(data.x[:,0],'m-',label='True data') 
    plt.grid()
    plt.legend()
    plt.title("Filtered IMM Continuous State Estimates, RC Circuit (PHA)")
    #plt.set_fontsize(20)   
 
#    plt.subplot(212)
#    plt.plot(min_error_filtered_m[0:210],'r.',label='Filtered mode assignment')
#    plt.plot(data.m[0:210],'b.',label='True Modes')
#    plt.grid()
#    plt.title('Mislabeled Modes, RC Circuit (PHA)')    
#    plt.legend()
    plt.show(block=False)

#    plt.figure()
#    plt.subplot(211)
#    plt.plot(sensor_data.x[200:215,0],'b.',label='Noisy data')
#    plt.plot(filtered_x[200:215,0],'r.',label='Filtered data')
#    plt.plot(data.x[200:215,0],'m-',label='True data') 
#    plt.grid()
#    plt.legend()
#    plt.title("Filtered IMM Continuous State Estimates (PHA)")
#    
#    plt.subplot(212)
#    plt.plot(mis_ind[200:215],'r.')
#    plt.grid()
#    plt.title('Mislabeled Modes')    
#    plt.show(block=False)
#
    #system.x = data.x
    #system.u = data.u
    mm_svm.svm_plot(system)
    plt.show(block=False)
    return cFilter, data

def learn_jmls():
    [system, data, noisy_data, junk1, junk2] = ts.rc_circuit_em(1000, 0.02, 0.01, True)
    print junk2
    with open('jmls_circuit.pkl', 'w') as output:
        pickle.dump(system,output)
        pickle.dump(data,output)
        pickle.dump(noisy_data, output) 
    return system, data, noisy_data
    
def test_jmls(sigmaR = [0.05], sigmaW = [0.05]):
    with open('jmls_circuit.pkl','r') as infile:
        system = pickle.load(infile)
        data = pickle.load(infile)
        noisy_data = pickle.load(infile)

    sensor_data = copy.deepcopy(noisy_data)
    Rd, Wd = set_W_R(system, data, sigmaR = sigmaR, sigmaW = sigmaW)
    cFilter = cPHA_Filter(system, Rd, Wd, sensor_data,data, True)
    filtered_x, filtered_m,min_error_filtered_m = cFilter.filter_system()
    
    plt.figure()
#    plt.subplot(211)
    plt.plot(sensor_data.x[:,0],'b.',label='Sensor data')
    plt.plot(filtered_x[:,0],'r.',label='Filtered data')
    plt.plot(data.x[:,0],'m-',label='True data') 
    plt.grid()
    plt.legend()
    plt.title("Filtered IMM Continuous State Estimates, RC Circuit (JMLS)")
    
#    plt.subplot(212)
#    plt.plot(min_error_filtered_m[0:210],'r.',label='Filtered mode assignment')
#    plt.plot(data.m[0:210],'b.',label='True Modes')
#    plt.grid()
#    plt.title('Mode Assignments, RC Circuit (JMLS)')    
#    plt.legend()
    plt.show(block=False)
#    
    return cFilter, data    

#def learn_racetrack():
#    [system, data, noisy_data, junk1, junk2] = ts.jet_route_fly_racetrack_em(5, 0.05, 0.005, False)
#    with open('em_racetrack.pkl', 'w') as output:
#        pickle.dump(system,output)
#        pickle.dump(data,output)
#        pickle.dump(noisy_data, output)
#    return 1
#
#def test_racetrack():
#    with open('em_racetrack.pkl','r') as infile:
#        system = pickle.load(infile)
#        data = pickle.load(infile)
#        noisy_data = pickle.load(infile)
#    sensor_data = copy.deep_copy(noisy_data)
#    Rd, Wd = set_W_R(0.05, system, data)
#    cFilter = cPHA_Filter(system, Rd, Wd, sensor_data,data, False)
#    filtered_x, filtered_m = cFilter.filter_system()
#    
#    plt.figure()
#    plt.subplot(211)
#    plt.plot(sensor_data.x[:,0], sensor_data.x[:,2],'b.',label='Sensor data')
#    plt.plot(filtered_x[:,0], filtered_x[:,2],'r.',label='Filtered data')
#    #plt.plot(simDataDisturbed.x[:,0],'m-',label='Dist. Sim. data') 
#    plt.grid()
#    plt.legend()
#    plt.title("Model fit")
#    plt.subplot(212)
#
#    plt.show(block=False)
#    return np.array([cFilter, data])
#
#def learn_racetrack_jmls():
#    [system, data, noisy_data, junk1, junk2] = ts.jet_route_fly_racetrack_em(5, 0.05, 0.005, True)
#    with open('jmls_racetrack.pkl', 'w') as output:
#        pickle.dump(system,output)
#        pickle.dump(data,output)
#        pickle.dump(noisy_data, output)
#    return 1
#
#def test_racetrack_jmls():
#    with open('em_racetrack.pkl','r') as infile:
#        system = pickle.load(infile)
#        data = pickle.load(infile)
#        noisy_data = pickle.load(infile)
#    sensor_data = copy.deepcopy(noisy_data)
#    Rd, Wd = set_W_R(0.05, system, data)
#    cFilter = cPHA_Filter(system, Rd, Wd, sensor_data,data, True)
#    filtered_x, filtered_m = cFilter.filter_system()
#    
#    plt.figure()
#    plt.plot(sensor_data.x[:,0], sensor_data.x[:,2],'b.',label='Sensor data')
#    plt.plot(filtered_x[:,0], filtered_x[:,2],'r.',label='Filtered data')
#    #plt.plot(simDataDisturbed.x[:,0],'m-',label='Dist. Sim. data') 
#    plt.grid()
#    plt.legend()
#    plt.title("Model fit")
#    plt.show(block=False)
#    return np.array([cFilter, data])
#
def learn_random():
    [system, data, noisy_data, junk1, junk2] = ts.jet_route_fly_random_em(17, 0.05, 0.005, False, svm_mis_penalty=1)
    print junk2
    with open('em_random.pkl', 'w') as output:
        pickle.dump(system,output)
        pickle.dump(data,output)
        pickle.dump(noisy_data, output)
    
    return system, data, noisy_data

def test_random(sigmaR = [0.7, 0.2, 0.7, 0.2], sigmaW = [0.025, 0.05, 0.025, 0.05]):
    with open('em_random.pkl','r') as infile:
        system = pickle.load(infile)
        data = pickle.load(infile)
        noisy_data = pickle.load(infile)

    sensor_data = copy.deepcopy(noisy_data)
    #Rd, Wd = set_W_R(0.05, system, data)
    Rd, Wd = set_W_R(sigmaR = sigmaR, sigmaW = sigmaW, system = system, data = data)
    cFilter = cPHA_Filter(system, Rd, Wd, sensor_data,data, False)
    filtered_x, filtered_m,min_error_filtered_m = cFilter.filter_system()
    plt.figure()
    #plt.subplot(211)
    plt.plot(sensor_data.x[0:465,0], sensor_data.x[0:465,2],'b.',label='Sensor data')
    plt.plot(filtered_x[0:465,0], filtered_x[0:465,2],'r.',label='Filtered data')
    plt.plot(data.x[0:465,0], data.x[0:465,2],'m-',label='True Data')
    #plt.plot(simDataDisturbed.x[:,0],'m-',label='Dist. Sim. data') 
    plt.grid()
    plt.legend()
    plt.title('Filtered IMM Continuous State Estimates, Random Pattern (PHA)')

#    plt.subplot(212)
#    plt.plot(filtered_m, 'r.', label='Filtered m')
#    plt.plot(data.m, 'g.', label='True m')
#    plt.title('Mode Estimates (PHA)')
#    plt.show(block=False)
#    
    plt.figure()
    plt.title('Velocities and Modes (PHA)')
    plt.subplot(211)
    plt.title('Vx progression, Random Pattern (PHA)')
    plt.plot(sensor_data.x[0:465,1],'b.',label='Sensor vx')
    plt.plot(filtered_x[0:465,1],'r.',label='Filtered vx')
    #plt.plot(data.x[0:465,1],'m-',label='True vx')
    plt.legend()

    plt.subplot(212)
    plt.title('Vy progression, Random Pattern (PHA)')
    plt.plot(sensor_data.x[0:465,3],'b.',label='Sensor vy')
    plt.plot(filtered_x[0:465,3],'r.',label='Filtered vy')
    #plt.plot(data.x[0:465,3],'m-',label='True vy')
    plt.legend()

#    plt.subplot(313)
#    plt.title('Mode Progression, Random Pattern (PHA)')
#    plt.plot(min_error_filtered_m[0:465],'r.',label='Filtered mode assignment')
#    plt.plot(data.m[0:465],'b.',label='True Modes')
#    plt.legend()
#   
    plt.show(block=False)
     
    return np.array([cFilter, data])

def learn_random_jmls():
    [system, data, noisy_data, junk1, junk2] = ts.jet_route_fly_random_em(17, 0.05, 0.005, True)
    with open('jmls_random.pkl', 'w') as output:
        pickle.dump(system,output)
        pickle.dump(data,output)
        pickle.dump(noisy_data, output)
    print junk2
    return [system, data, noisy_data]


def test_random_jmls(sigmaR = [0.7, 0.2, 0.7, 0.2], sigmaW = [0.025, 0.05, 0.025, 0.05]):
    with open('jmls_random.pkl','r') as infile:
        system = pickle.load(infile)
        #data = pickle.load(infile)
        #noisy_data = pickle.load(infile)

    with open('em_random.pkl', 'r') as infile:
        junk = pickle.load(infile)
        data = pickle.load(infile)
        noisy_data = pickle.load(infile)

    sensor_data = copy.deepcopy(noisy_data)
    Rd, Wd = set_W_R(sigmaR = sigmaR, sigmaW = sigmaW, system = system, data = data)
    cFilter = cPHA_Filter(system, Rd, Wd, sensor_data,data, False)
    filtered_x, filtered_m,min_error_filtered_m = cFilter.filter_system()

   
    plt.figure()
    #plt.subplot(211)
    plt.plot(sensor_data.x[0:465,0], sensor_data.x[0:465,2],'b.',label='Sensor data')
    plt.plot(filtered_x[0:465,0], filtered_x[0:465,2],'r.',label='Filtered data')
    plt.plot(data.x[0:465,0], data.x[0:465,2],'m-',label='True data')
    #plt.plot(simDataDisturbed.x[:,0],'m-',label='Dist. Sim. data') 
    plt.grid()
    plt.legend()
    plt.title('Filtered IMM Continuous State Estimates, Random Pattern (JMLS)')

 #    plt.subplot(212)
 #    plt.plot(filtered_m, 'r.', label='Filtered m')
 #    plt.plot(data.m, 'g.', label='True m')
 #    plt.title('Mode Estimates (JMLS)')
 #    plt.show(block=False)
     
    plt.figure()
    plt.title('Velocities and Modes (JMLS)')
    plt.subplot(211)
    plt.title('Vx progression, Random Pattern (JMLS)')
    plt.plot(sensor_data.x[0:465,1],'b.',label='Sensor vx')
    plt.plot(filtered_x[0:465,1],'r.',label='Filtered vx')
    #plt.plot(data.x[0:465,1],'m-',label='True vx')
    plt.legend()

    plt.subplot(212)
    plt.title('Vy progression, Random Pattern (JMLS)')
    plt.plot(sensor_data.x[0:465,3],'b.',label='Sensor vy')
    plt.plot(filtered_x[0:465,3],'r.',label='Filtered vy')
    #plt.plot(data.x[0:465,3],'m-',label='True vy')
    plt.legend()

#    plt.subplot(313)
#    plt.title('Mode Progression, Random Pattern (JMLS)')
#    plt.plot(min_error_filtered_m[0:465],'r.',label='Filtered mode assignment')
#    plt.plot(data.m[0:465],'b.',label='True Modes')
#    plt.legend()
#
    plt.show(block=False)  
    
    return np.array([cFilter, data])

def learn_lawnmower():
    [system, data, noisy_data, junk1, junk2] = ts.jet_route_fly_lawnmower_em(is_jmls = False)
    print junk2
    with open('em_lawnmower.pkl', 'w') as output:
        pickle.dump(system,output)
        pickle.dump(data,output)
        pickle.dump(noisy_data, output)
    
    return system, data, noisy_data

def test_lawnmower(sigmaW = [0.005, 0.01, 0.005, 0.01]):
    with open('em_lawnmower.pkl','r') as infile:
        system = pickle.load(infile)
        data = pickle.load(infile)
        noisy_data = pickle.load(infile)

    sensor_data = copy.deepcopy(noisy_data)
    #Rd, Wd = set_W_R(0.05, system, data)
    Rd, Wd = set_W_R(sigmaR = [0.8, 0.3, 0.8, 0.3], sigmaW=sigmaW, system = system, data = data)
    cFilter = cPHA_Filter(system, Rd, Wd, sensor_data,data, False)
    filtered_x, filtered_m,min_error_filtered_m = cFilter.filter_system()
    
    plt.figure()
    #plt.subplot(211)
    plt.plot(sensor_data.x[:,0], sensor_data.x[:,2],'b.',label='Sensor data')
    plt.plot(filtered_x[:,0], filtered_x[:,2],'r.',label='Filtered data')
    plt.plot(data.x[:,0], data.x[:,2],'m-',label='True Data')
    #plt.plot(simDataDisturbed.x[:,0],'m-',label='Dist. Sim. data') 
    plt.grid()
    plt.legend()
    plt.title('Filtered IMM Continuous State Estimates, Lawnmower Pattern (PHA)')

#    plt.subplot(212)
#    plt.plot(filtered_m, 'r.', label='Filtered m')
#    plt.plot(data.m, 'g.', label='True m')
#    plt.title('Mode Estimates (PHA)')
#    plt.show(block=False)
#    
    plt.figure()
    plt.title('Velocities and Modes (PHA)')
    plt.subplot(211)
    plt.title('Vx progression, Lawnmower Pattern (PHA)')
    plt.plot(sensor_data.x[0:465,1],'b.',label='Sensor vx')
    plt.plot(filtered_x[0:465,1],'r.',label='Filtered vx')
    #plt.plot(data.x[0:465,1],'m-',label='True vx')
    plt.legend()

    plt.subplot(212)
    plt.title('Vy progression, Lawnmower Pattern (PHA)')
    plt.plot(sensor_data.x[0:465,3],'b.',label='Sensor vy')
    plt.plot(filtered_x[0:465,3],'r.',label='Filtered vy')
    #plt.plot(data.x[0:465,3],'m-',label='True vy')
    plt.legend()

#    plt.subplot(313)
#    plt.title('Mode Progression, Lawnmower Pattern (PHA)')
#    plt.plot(min_error_filtered_m[0:465],'r.',label='Filtered mode assignment')
#    plt.plot(data.m[0:465],'b.',label='True Modes')
#    plt.legend()
#   
    plt.show(block=False)
    return np.array([cFilter, data])

def learn_lawnmower_jmls():
    [system, data, noisy_data, junk1, junk2] = ts.jet_route_fly_lawnmower_em(is_jmls = True)
    print junk2
    with open('jmls_lawnmower.pkl', 'w') as output:
        pickle.dump(system,output)
        pickle.dump(data,output)
        pickle.dump(noisy_data, output)
    return [system, data, noisy_data]


def test_lawnmower_jmls(sigmaW = [0.005, 0.01, 0.005, 0.01]):
    with open('jmls_lawnmower.pkl','r') as infile:
        system = pickle.load(infile)
        data = pickle.load(infile)
        noisy_data = pickle.load(infile)

    sensor_data = copy.deepcopy(noisy_data)
    Rd, Wd = set_W_R(sigmaR = [0.8,0.3,0.8,0.3], sigmaW = sigmaW, system = system, data = data)

    cFilter = cPHA_Filter(system, Rd, Wd, sensor_data,data, False)
    filtered_x, filtered_m,min_error_filtered_m = cFilter.filter_system()
    
    plt.figure()
    #plt.subplot(211)
    plt.plot(sensor_data.x[0:465,0], sensor_data.x[0:465,2],'b.',label='Sensor data')
    plt.plot(filtered_x[0:465,0], filtered_x[0:465,2],'r.',label='Filtered data')
    plt.plot(data.x[0:465,0], data.x[0:465,2],'m-',label='True data')
    #plt.plot(simDataDisturbed.x[:,0],'m-',label='Dist. Sim. data') 
    plt.grid()
    plt.legend()
    plt.title('Filtered IMM Continuous State Estimates (JMLS)')

#    plt.subplot(212)
#    plt.plot(filtered_m, 'r.', label='Filtered m')
#    plt.plot(data.m, 'g.', label='True m')
#    plt.title('Mode Estimates (JMLS)')
#    plt.show(block=False)
    
    plt.figure()
    plt.title('Velocities and Modes (JMLS)')
    plt.subplot(211)
    plt.title('Vx progression, Lawnmower Pattern (JMLS)')
    plt.plot(sensor_data.x[0:465,1],'b.',label='Sensor vx')
    plt.plot(filtered_x[0:465,1],'r.',label='Filtered vx')
    #plt.plot(data.x[0:465,1],'m-',label='True vx')
    plt.legend()

    plt.subplot(212)
    plt.title('Vy progression, Lawnmower Pattern (JMLS)')
    plt.plot(sensor_data.x[0:465,3],'b.',label='Sensor vy')
    plt.plot(filtered_x[0:465,3],'r.',label='Filtered vy')
    #plt.plot(data.x[0:465,3],'m-',label='True vy')
    plt.legend()

#    plt.subplot(313)
#    plt.title('Mode Progression, Lawnmower Pattern (JMLS)')
#    plt.plot(min_error_filtered_m[0:465],'r.',label='Filtered mode assignment')
#    plt.plot(data.m[0:465],'b.',label='True Modes')
#    plt.legend()
#
    plt.show(block=False)
    return np.array([cFilter, data])



if __name__ == "__main__": 
    test_lawnmower()                               
