import numpy as np
import numpy.linalg as la
import filtering as flt
import pdb

class EKF(flt.Stochastic_Filter):
    """Extended Kalman Filter (EKF) class.
        pf: pointer to time propagation function.
        pl: pointer to model linearization function (Jacobian).
        mf: pointer to measurement function.
        ml: pointer to measurement linearization function (Jacobian)
    """
    def __init__(self,pf,pl,mf,ml):
        super(EKF,self).__init__()
        self._propag_func = pf
        self._process_lin = pl
        self._meas_func = mf
        self._meas_lin = ml        
        self._last_innov = np.array([]) #Last inovation
        
    def prediction_step(self,u):    
        """Prediction step given input vector."""
        x_prev = self.get_x()
        Q_prev = self.get_Q()
        Wd = self.get_Wd()
        pf = self.get_propag_func()
        pl = self.get_process_lin_func()
                
        x_pred,Q_pred = self._EKF_prediction(u,x_prev,Q_prev,Wd,pf,pl)       
        
        self.set_x_pred(x_pred)        
        self.set_Q_pred(Q_pred)
        self.set_x(x_pred)      
        self.set_Q(Q_pred)      
        
        
    def correction_step(self,y):    
        """Correction step given measurement vector."""
        x_pred = self.get_x()
        Q_pred = self.get_Q()
        Rd = self.get_Rd()
        mf = self.get_meas_func()
        ml = self.get_meas_lin_func()
        
        x_up,Q_up = self._EKF_update(y,x_pred,Q_pred,Rd,mf,ml)
        
        self.set_x_up(x_up)        
        self.set_Q_up(Q_up)
        self.set_x(x_up)      
        self.set_Q(Q_up) 
        
    def _EKF_prediction(self,u,x_prev,Q_prev,Wd,propag_func,process_lin):
        """General EKF prediction step."""
        x_predicted = propag_func(x_prev,u)
        A_jacob = process_lin(x_prev,u)        
        Q_predicted = A_jacob.dot(Q_prev).dot(A_jacob.transpose())+Wd         
        return x_predicted,Q_predicted

    def _EKF_update(self,y,x_pred,Q_pred,Rd,meas_func,meas_lin):   
        """General EKF correction step."""
        y_pred = meas_func(x_pred)
        C_jacob = meas_lin(x_pred)
       	   
        cov_meas = C_jacob.dot(Q_pred).dot(C_jacob.transpose())+Rd
        #pdb.set_trace()
        L = Q_pred.dot(C_jacob.transpose()).dot(la.inv(cov_meas))        
        
        self._last_innov = y-y_pred                 
        x_up = x_pred+L.dot(self._last_innov)

        D = np.eye(x_pred.size)-L.dot(C_jacob)
        
        Q_up = D.dot(Q_pred).dot(D.transpose())+L.dot(Rd).dot(L.transpose())
        Q_up = (Q_up+Q_up.transpose())/2.0	
        
        return x_up,Q_up

    def get_last_innovation(self):
        """Last innovation (y-y_pred) computed by the filter."""
        return self._last_innov

    def get_propag_func(self):
        """Pointer to the time propagation function."""
        return self._propag_func
    
    def get_process_lin_func(self):
        """Pointer to process linearization function (Jacobian)."""
        return self._process_lin

    def get_meas_func(self):
        """Pointer to measurement function."""
        return self._meas_func
    
    def get_meas_lin_func(self):
        """Pointer to measurement linearization function (Jacobian)."""
        return self._meas_lin
