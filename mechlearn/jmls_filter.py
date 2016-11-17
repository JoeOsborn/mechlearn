import EKF_filtering
import IMM_filtering
import numpy as np
import testing
import pdb
import pickle
import numpy.linalg as la
import jmls_data
import matplotlib.pyplot as plt

class jmls_Filter:
    def __init__(self, system, data, true):
        self.data = data
        self.system = system
        self.true_data = true

    def filter_system(self):
        #Initialize filter_list
        filter_list = np.array([])

        for i in range(self.system.K):
            m = int(i)
            pf = lambda x_prev, u, m = i: self.propag_function(m, x_prev, u)
            pl = lambda x_prev, u, m = i: self.propag_lin(m, x_prev, u)
            mf = lambda x_prev: self.measure_function(x_prev)
            ml = lambda x_prev: self.measure_lin(x_prev)
            #pdb.set_trace()
            new_filter = EKF_filtering.EKF(pf, pl, mf, ml)
            new_filter.init(self.data.x[0], self.data.Qm[i])
            new_filter.set_Rd(self.data.Qm[i])
            new_filter.set_Wd(np.zeros(np.shape(self.data.Qm[i])))
            filter_list = np.append(filter_list, new_filter)

        #Initialize data lists
        filtered_x = np.zeros(self.data.x.shape)
        filtered_m = np.zeros(self.data.m.shape)
    
        #Initialize IMM Filter

        TPM = np.identity(self.system.K)
        #Update TPM
        for j in  range(self.system.K):
            TPM[j] = self.system.ptrans[j, 0, :]

        imm_filter = IMM_filtering.IMM_Filter(filter_list, TPM)
        filtered_x[0] = self.data.x[0]

        #Run IMM Filter
        for i in range(1, len(self.data.m)):
            #Call filter to estimate current X
            if len(self.data.u) > 0:
                imm_filter.estimate(self.data.u[i], self.data.x[i])
            else:
                imm_filter.estimate(None, self.data.x[i])     
    
            #Mark Data
            filtered_x[i] = imm_filter.get_x_IMM()
            filtered_m[i] = np.argmax(imm_filter.get_mode_prob_vector())


        #Calculate RMSE
        rmse = la.norm(filtered_x - self.true_data.x)/np.sqrt(len(self.data.m))

        #Calculate Misclassification rate
        missclassified = np.sum([1 for diff in filtered_m - self.true_data.m if diff != 0])

        #return np.array([RMSE
        print rmse
        print missclassified
        return filtered_x

    def propag_function(self, m, x_prev, u):
        x = None
        if len(u) > 0:
            x = np.dot(self.system.Am[m],x_prev) + np.dot(self.system.Bm[m],u)
        else:
            x = np.dot(self.system.Am[m],x_prev)

        return x

    def propag_lin(self, m, x_prev, u):
        jacobian = np.copy(self.system.Am[m])   
    
        #Add the effects of Bu
        if len(u) > 0:
            for i in range(len(x_prev)):
                jacobian[:, i] += np.dot(self.system.Bm[m],u)

        #Add the effects of the rest of the x elements
        for i in range(len(x_prev)): 
            x_temp = np.copy(x_prev)
            x_temp[i] = 0
            jacobian[:, i] += np.dot(self.system.Am[m],x_temp)
    
        return jacobian

    def measure_function(self, x_prev):
        return x_prev

    def measure_lin(self, x_prev):
        return np.identity(len(x_prev))


def learn():
    [system, data, noisy_data, junk1, junk2] = jmls_data.jmls_rc_circuit(1000)
    with open('jmls.pkl', 'w') as output:
        pickle.dump(system,output)
        pickle.dump(data,output)
        pickle.dump(noisy_data, output)

def test():
    #[system, data, junk1, junk2] = testing.test_em(1000)
    with open('jmls.pkl','r') as infile:
        system = pickle.load(infile)
        data = pickle.load(infile)
        noisy_data = pickle.load(infile)
    cFilter = jmls_Filter(system, noisy_data, data)
    filtered_x = cFilter.filter_system()

    plt.figure()
    plt.plot(noisy_data.x[:,0],'b.',label='Noisy data')
    plt.plot(filtered_x[:,0],'r.',label='Filtered data')
    #plt.plot(simDataDisturbed.x[:,0],'m-',label='Dist. Sim. data') 
    plt.grid()
    plt.legend()
    plt.title("Model fit")
    plt.show(block=False)
                                
    return np.array([cFilter, data])
