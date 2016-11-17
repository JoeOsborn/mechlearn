import numpy as np
import numpy.linalg as la
#from scipy.stats import multivariate_normal as mvn
import filtering as flt
import pdb 
import IPython

class IMM_Filter(flt.MM_Filter):
    """Interacting Multiple Models (IMM) filter implementation."""
    #fTPM is a function pointer that returns the transition probability matrix
    #Takes current x and u as input
    def __init__(self, system, data, filter_list, fTPM):
        super(IMM_Filter,self).__init__(filter_list)
        #Parameter extraction (efficiency )
        self._num_models = self.get_num_models() 
        self._dim_state = self.get_filter(0).get_x().size
        
        self._fTPM = fTPM #Transition probability matrix
        self._data = data
        self._system = system
        self._mixed_state_vectors = [np.array([])]*self._num_models
        self._mixed_cov_matrices = [np.array([])]*self._num_models
        
        self._mode_prob_vec = np.ones(self._num_models)*(1.0/self._num_models)
        self._mode_prob_vec_pred = np.copy(self._mode_prob_vec)
        self._mode_prob_vec_up = np.copy(self._mode_prob_vec)
        
        self._x_imm = np.array([])        
        self._Q_imm = np.array([])        
        
        self._curr_index = 0

    def get_next_estimate(self):
        x_data = self._data.x[self._curr_index]
        if len(self._data.u) > 0:
            u_data = self._data.u[self._curr_index]
        else:
            u_data = None

        self._TPM = self._fTPM(u_data, x_data)
        self._curr_index += 1

        self.estimate(u_data, x_data)

        x = self.get_x_IMM()
        m = np.argmax(self.get_mode_prob_vector())
        #print 'X: ' + str(x)
        #print 'm: ' + str(m)
        return x, m

    def estimate(self,u,y):    
        """Full estimation step given input and observation vectors.""" 
        self._mode_prob_prediction()
        self._mixed_state_vectors,self._mixed_cov_matrices = self._mix_estimates()        
        self._run_filter_bank(u,y)        
        self._pred_state_list,self._pred_cov_list = self._get_predicted_estimates_list()        
        self._corr_state_list, self._corr_cov_list = self._get_corrected_estimates_list()        
        #print "Corrected State List: " + str(self._corr_state_list)
        #print "Predicted_state_list: " + str(self._pred_state_list)
        #print self._corr_state_list, self._corr_cov_list, self._pred_state_list,self._pred_cov_list
        self._mode_prob_update(y)        
        return self._imm_output()
    
    def _mode_prob_prediction(self):
        """Mode probability prediction using Markovian model. Assumes 
        column->row disposition of the probabilities in the TPM."""        
        self._mode_prob_vec_pred = self._TPM.dot(self._mode_prob_vec)
        self._mode_prob_vec_pred/= np.sum(self._mode_prob_vec_pred) #Normalize
        self._mode_prob_vec = self._mode_prob_vec_pred
        #print self._mode_prob_vec
        #print sef._TPM
    
    def _mix_estimates(self):
        """Mixes state and covariance estimates together, which are subsequently
        used as initial conditions for the bank of stochastic filters"""
        curr_states,curr_covs = self.get_estimates_list()
        curr_mode_prob = self.get_mode_prob_vector()
        pred_mode_prob = self._get_predicted_mode_prob_vector()
        TPM = self.get_TPM()
        
        mixed_states=[]; mixed_covs=[]
        
        #Mixes states
        for i in range(self._num_models):
            mixed_states.append(np.zeros(self._dim_state))            
            for j in range(self._num_models):
                mixed_states[i] += TPM[i,j]*curr_mode_prob[j]*curr_states[j]            
            #if pred_mode_prob[i] < 0.001:
            #    pdb.set_trace()
            if pred_mode_prob[i]>0.01:
                mixed_states[i] = mixed_states[i]/pred_mode_prob[i]
            else:
                mixed_states[i] = np.zeros(self._dim_state)
                print "Error: Zero probability when mixing states"
           
           

        #Mixes covariances
        for i in range(self._num_models):
            mixed_covs.append(np.zeros([self._dim_state,self._dim_state]))            
            
            for j in range(self._num_models):
                diff = curr_states[j]-mixed_states[i] 
                #pdb.set_trace()
                mixed_covs[i] += TPM[i,j]*curr_mode_prob[j]*(curr_covs[j]+np.outer(diff,diff))
            
            if pred_mode_prob[i]>0.01:
                mixed_covs[i] = mixed_covs[i]/pred_mode_prob[i]
            else:
                mixed_covs[i]=np.zeros([self._dim_state,self._dim_state])
                print "Error: Zero probability when mixing covariances"       

          
        return mixed_states,mixed_covs
    
    def _run_filter_bank(self,u,y):
        for i in range(self._num_models):            
            #Sets the mixed initial conditions
            stoch_filter = self.get_filter(i)
            stoch_filter.set_x(self._get_mixed_state(i))
            stoch_filter.set_Q(self._get_mixed_covariance(i))            
            #Prediction and correction steps for the current filter
            stoch_filter.prediction_step(u)
            stoch_filter.correction_step(y)
    
    def _mode_prob_update(self,y):
        """Updates the mode probability vector using the measurements."""
        innov_list = self.get_last_innovation_list()
        power = np.zeros([self._num_models])
        sqrt_det_R = np.zeros([self._num_models])
        likelihood_vector = np.zeros((self._num_models))
        dim_output = y.size
        
        for i in range(self._num_models):
            innov = innov_list[i] #Innovation for i-th filter                        
            R = self.get_filter(i).get_Rd() #Sensor noise
            if(R.size > 1):
                power[i] = np.exp(-0.5*np.dot(innov.transpose(),la.inv(R)).dot(innov))
                sqrt_det_R[i] = np.sqrt(la.det(R))
            else:
                power[i] = np.exp(-0.5*innov.transpose()*(1.0/R)*innov)
                sqrt_det_R[i] = np.sqrt(R)
            #pdb.set_trace()
            C = ((2.0*np.pi)**(dim_output/2.0))*sqrt_det_R[i]
            likelihood_vector[i] = power[i]/C
        
       # pdb.set_trace()
        for i in range(self._num_models): 
            self._mode_prob_vec_up[i] = (self._mode_prob_vec_pred[i]/sqrt_det_R[i])*power[i]
        
        prob_sum = np.sum(self._mode_prob_vec_up)
           
        if prob_sum > 0.0:
            self._mode_prob_vec_up/=prob_sum
        else:
            #pdb.set_trace()
            print "ERROR: Zero probability vector for mode probabilities."
            self._mode_prob_vec_up = np.copy(self._mode_prob_vec_pred)
                    
        self._mode_prob_vec = self._mode_prob_vec_up

    def _imm_output(self):
        """Generates averaged state and covariance estimates."""
        x_IMM = np.zeros([self._dim_state])
        Q_IMM = np.zeros([self._dim_state,self._dim_state])
    
        #Average state estimate
        for i in range(self._num_models):
            x_IMM += self._mode_prob_vec_up[i]*self._corr_state_list[i]

        #Average covariance estimate
        for i in range(self._num_models):
            prob = self._mode_prob_vec_up[i]
            diff = self._corr_state_list[i]-x_IMM
            cov = self._corr_cov_list[i]                        
            Q_IMM += prob*(cov+np.outer(diff,diff))
    
        self._x_imm = x_IMM; self._Q_imm = Q_IMM
    
        return x_IMM,Q_IMM,self.get_mode_prob_vector()
    
    def get_mode_prob_vector(self):
        return self._mode_prob_vec
    
    def _get_predicted_mode_prob_vector(self):
        return self._mode_prob_vec_pred
    
    def _get_corrected_mode_prob_vector(self):
        return self._mode_prob_vec_up
    
    def get_x_IMM(self):
        """Returns IMM's average state estimate."""
        return self._x_imm
    
    def get_Q_IMM(self):
        """Returns IMM's average covariance estimate."""
        return self._Q_imm
    
    def get_TPM(self):
        """Transition Probability Matrix for the Markovian mode transitions."""
        return self._TPM
    
    def _get_mixed_state(self,index):
        """Returns a particular mixed state estimate."""
        return self._mixed_state_vectors[index]
       
    def _get_mixed_covariance(self,index):
        """Returns a particular mixed covariance estimate."""
        return self._mixed_cov_matrices[index]

    def get_estimates_list(self):
        """Returns a list of current best estimates."""
        state_list = []; cov_list = []
        for filt in self.get_filter_list():
            state_list.append(filt.get_x())
            cov_list.append(filt.get_Q())
        return state_list,cov_list
    
    def get_last_innovation_list(self):
        """List of all last innovation terms computed by the filters."""                
        return [filt.get_last_innovation() for filt in self.get_filter_list()]
    
    def _get_predicted_estimates_list(self):
        """Returns a list of predicted estimates."""
        state_list = []; cov_list = []
        for filt in self.get_filter_list():
            state_list.append(filt.get_x_pred())
            cov_list.append(filt.get_Q_pred())        
        return state_list,cov_list
            
    def _get_corrected_estimates_list(self):
        """Returns a list of corrected estimates."""
        state_list = []; cov_list = []
        for filt in self.get_filter_list():
            state_list.append(filt.get_x_up())
            cov_list.append(filt.get_Q_up())        
        return state_list,cov_list       
   
    def set_TPM(self,TPM):
        """Sets the TPM"""
        self._TPM = TPM
    
    
    
    
    

