import numpy as np
import pdb

class Stochastic_Filter(object):
    """Base class for all stochastic filters."""
        
    _id_counter = 0    #static ID counter for all instantiated filters

    def __init__(self):
        Stochastic_Filter._id_counter +=1
        self._id = Stochastic_Filter._id_counter

        self._x = np.array([])
        self._x_up = np.array([])
        self._x_pred = np.array([])
        
        self._Q = np.array([])
        self._Q_up = np.array([])
        self._Q_pred = np.array([])
    
        self._Wd = np.array([])
        self._Rd = np.array([])
    
    def init(self,x0,Q0):
        #pdb.set_trace()
	"""Sets all initial conditions."""
        self.set_x(np.copy(x0))
        self.set_Q(np.copy(Q0))
        self.set_x_pred(np.zeros(x0.shape))
        self.set_Q_pred(np.zeros(Q0.shape))
        self.set_x_up(np.zeros(x0.shape))
        self.set_Q_up(np.zeros(Q0.shape))

    def prediction_step(self,u):    
        """Prediction step given input vector."""
        print "\nWARNING: Purely virtual function call."
    
    def correction_step(self,y):    
        """Correction step given measurement vector."""
        print "\nWARNING: Purely virtual function call."
    
    def get_id(self):
        """Unique ID for the filter object."""
        return self._id

    def get_x(self):
        """Current state estimate."""
        return self._x
    
    def get_x_up(self):
        """Updated state estimate."""
        return self._x_up
    
    def get_x_pred(self):
        """Predicted state estimate."""
        return self._x_pred
    
    def get_Q(self):
        """Current covariance estimate."""
        return self._Q
    
    def get_Q_up(self):
        """Updated covariance estimate."""
        return self._Q_up
    
    def get_Q_pred(self):
        """Predicted covariance estimate."""
        return self._Q_pred

    def get_Wd(self):
        """Process covariance matrix."""
        return self._Wd
    
    def get_Rd(self):
        """Measurement covariance matrix."""
        return self._Rd

    def set_x(self,x):
        """Sets the current state estimate."""
        self._x = x
    
    def set_x_up(self,x_up):
        """Sets the updated state estimate."""
        self._x_up = x_up
    
    def set_x_pred(self,x_pred):
        """Sets the predicted state estimate."""
        self._x_pred = x_pred
    
    def set_Q(self,Q):
        """Sets the current covariance estimate."""
        self._Q = Q
    
    def set_Q_up(self,Q_up):
        """Sets the updated covariance estimate."""
        self._Q_up = Q_up
    
    def set_Q_pred(self,Q_pred):
        """Sets the predicted covariance estimate."""
        self._Q_pred = Q_pred

    def set_Wd(self,Wd):
        """Sets the process covariance matrix."""
        self._Wd = Wd
    
    def set_Rd(self,Rd):
        """Sets the measurement covariance matrix."""
        self._Rd = Rd



class MM_Filter(object):
    """Base class for implementing multiple model filters"""
    def __init__(self,filter_list):
        self._filter_list = filter_list
    
    def estimate(self,u,y):    
        """Full estimation step given input and observation vectors.""" 
        print "\nWARNING: Purely virtual function call."    
    
    def get_filter_list(self):
        """List of filters used to track the multiple models"""
        return self._filter_list
    
    def get_filter(self,index):
        """Retrieves an specific filter object given its index."""
        return self._filter_list[index]
            
    def get_num_models(self):
        """Total number of models"""
        return len(self._filter_list)
        
    def set_filter_list(self,filter_list):
        """Sets the list of filters."""
        self._filter_list = filter_list
    
    def set_filter(self,index,filter_ob):
        """Sets an specific filter object given its index."""
        self._filter_list[index] = filter_ob
    
    
    
    
    
    
    
    
    
    
    
    
    
