import numpy as np
import math
import copy
import pdb
import random

class MMData:
    def __init__(self):
        self.x = np.array([])   
        self.u = np.array([])   
        self.m = np.array([])
        self.Qm= []
        self.trans_index = [] 
        self.num_models=1
        self.R=[]   
    def num_x_samples(self):
        if self.x.size == 0:
            return 0
        else:
            return self.x.shape[0]
    
    def num_u_samples(self):
        if self.u.size == 0:
            return 0
        else:
            return self.u.shape[0]

    def get_transition_indices(self):       
        self.trans_index = []
        nm = self.m.size        
        if nm> 0: #potentially multiple models                      
            for i in range(1,nm,1):
                if(self.m[i-1] != self.m[i]):                                       
                    self.trans_index.append(i-1) #Stores index of transition

class MMSystem:
    def __init__(self):
        self.Am=[]                      #Estimates of process matrices
        self.Bm=[]                      #Estimates of input matrices
        self.Qm=[]
        self.R = []                     #Estimates of noise covariance matrices     
        self.x=np.array([])         #Estimates of continuous state
        self.u=np.array([])             #input vector                       
        self.m=np.array([])         #Estimates of discrete modes
        self.pm1=np.array([])           #probability vector of initial mode
        self.ptrans = np.array([])      #guarded transition probabilities
        self.x1=np.array([])            #initial state vector       
        self.guards =[]     
        self.n=0
        self.K = 0                      #number of classes
        self.lik_norm=1.0               #factor for normalizing likelihoods
        self.lik_norm_flag=False        #likelihood normalization flag
        self.scale_alpha_beta = False   #flag for scaling alphas and betas
        self.scale_vector=np.array([])#Stores the scaling factors
        self.data = None                #MMData object
        self.W = np.array([])   
    def get_initial_state(self):        
        return self.x1  
    def set_initial_state(self,x1):
        self.x1= np.copy(x1)        
    def get_num_samples(self):
        return self.n   
    def get_num_modes(self):
        return len(self.Am)
    def calculate_W(self,true_data):
        W = np.zeros((true_data.x.shape[1], true_data.x.shape[1]))
        
        for i in range(true_data.x.shape[0] - 1):
            X = true_data.x[i]
            if len(self.Bm) > 0:
                A = self.Am[true_data.m[i]]
                B = self.Bm[true_data.m[i]]      #Model selection                                
                predX = np.dot(A,X)+np.dot(B,true_data.u[i,:])
            else:
                A = self.Am[true_data.m[i]]
                predX = np.dot(A,X) #Step   
            
            diff = np.abs(true_data.x[i+1] - predX)
            
            for j in range(W.shape[0]):
                if W[j,j] < diff[j]:
                    W[j,j] = diff[j]
        #pdb.set_trace()            
        #for i in range(W.shape[0]):
        #   W[i,i] = ((2.0/3.0)*W[i,i])**2
        self.W = W
                
        return W 


#Creates an MM system with all fields initialized, except for the model
#matrices
def system_from_data(data):
    system = MMSystem()
    #Reference to the data used to generate estimates
    system.data = copy.deepcopy(data) 
    
    #Makes the estimates equal to the measurements
    system.Qm = copy.deepcopy(data.Qm)
    system.R = copy.deepcopy(data.R)
    system.x = copy.deepcopy(data.x)
    system.u = copy.deepcopy(data.u)
    system.m = copy.deepcopy(data.m)    
    
    #Parameters
    system.set_initial_state(copy.deepcopy(data.x[0,:]))        
    system.n = data.x.shape[0]
    system.K = len(data.Qm)
    
    return system

def simulate_mm_system(system):
        
    simData = MMData()
    
    X = np.copy(system.get_initial_state())     
    dimx = X.size
    
    simData.x = np.zeros([system.n,dimx]);  
    simData.x[0,:] = X
    
    simData.u = np.copy(system.u)
    simData.m = np.copy(system.m)
            
    if((system.u.size == 0) or (len(system.Bm)==0)):
        print "System with no input detected."
                
        if((system.m.size == 0) or (len(system.Am)==1)):
            print "Single model system detected."
            
            simData.num_models=1
            
            A = system.Am[0]            
            for i in range(1,system.n,1):               
                X = np.dot(A,X)#Step
                simData.x[i,:] = X  
        else:
            print "Multiple model system detected."
            
            simData.num_models= len(system.Am)
            
            ind = 1             
            for i in range(system.m.size-1):        
                A = system.Am[system.m[i]]              
                X = np.dot(A,X) #Step
                simData.x[i+1,:] = X            
    else:
        print "System with inputs detected."
        
        if((system.m.size == 0) or (len(system.Am)==1)):
            print "Single model system detected."
        
            simData.num_models=1
        
            A = system.Am[0]
            B = system.Bm[0]                        
            for i in range(1,system.n,1):               
                X = np.dot(A,X)+np.dot(B,simData.u[i-1,:]) #Step
                simData.x[i,:] = X                      
        else:
            print "Multiple model system detected."
            
            simData.num_models= len(system.Am)
            
            ind = 1             
            for i in range(system.m.size-1):        
                A = system.Am[system.m[i]]
                B = system.Bm[system.m[i]]  #Model selection                
                X = np.dot(A,X)+np.dot(B,simData.u[ind-1,:]) #Step
                simData.x[ind,:] = X        
                ind+=1
    
    return simData
    

#
#def simple_linsystem(Nsamples):
#    A = np.array([[0.2, -0.3],[0.5,0.8]])
#    B = np.row_stack([1,-2])
#    X = np.row_stack([10,10])
#    
#    data = MMData()
#    
#    data.num_models = 1 
#    data.u = np.row_stack(np.arange(Nsamples)*0.02)
#    data.x = np.zeros([Nsamples,2]);
#
#    data.x[0,:] = X.transpose()
#
#    sigma = 0.03
#    data.Qm = [np.identity(2)*(sigma**2)]
#
#    for i in range(1,Nsamples,1):                   
#        X = np.dot(A,X)+np.dot(B,data.u[i-1,0])     
#        data.x[i,:] = (X+np.random.randn(2,1)*sigma).transpose()
#    
#    return data
#
#
##Adapted from [Seah, Hwang (2009)]
#def jet_route_fly(Nsamples,Ts, sigma):
#    
#    #X = [x1 x2] aircraft position in arbitrary coord. frame
#    
#    #Constant Velocity (CV)
#    Acv = np.identity(4)
#    Acv[0,1] = Acv[2,3] = Ts
#
#    #Coordinated turn
#    w = 1.5*math.pi/180.0;
#    Act = np.identity(4)
#    Act[0,1] = Act[2,3] = math.sin(w*Ts)/w
#    Act[0,3] =  (1-math.cos(w*Ts))/w
#    Act[2,1] = - Act[0,3]
#    Act[1,1] = Act[3,3] = math.cos(w*Ts)
#    Act[1,3] = -math.sin(w*Ts)
#    Act[3,1] = -Act[1,3]
#    
#    #Starts at the origin with nonzero velocity
#    X = np.array([0,2,0,2])
#
#    data = MMData()
#
#    #State vector samples
#    data.x = np.zeros([Nsamples,4])
#    data.x[0,:] = X
#    
#    #Model samples
#    data.num_models = 2
#    data.m = np.array([0]*(Nsamples))
#    
#    sigma = 0.01    #Process noise
#    data.Qm = [np.array([sigma**2]),np.array([sigma**2])]
#    
#    for i in range(1,Nsamples,1):                           
#                
#        if i <= math.ceil(Nsamples/2.0):
#            A = Acv
#            m = 0
#        else:
#            A = Act
#            m = 1
#        
#        X = np.dot(A,X) 
#        data.x[i,:] = X+np.random.randn(4)*sigma
#        
#        data.m[i] = m
#            
#    return data
#
#def jet_fly_racetrack(Nloops,Ts, sigma, sigmaLik):
#    
#    #X = [x1 x1' x2 x2'] aircraft position in arbitrary coord. frame
#    
#    #Constant Velocity (CV)
#    Acv = np.identity(4)
#    Acv[0,1] = Acv[2,3] = Ts
#
#    #Coordinated turn ccw
#    w = -10*math.pi/180.0;
#    Acw = np.identity(4)
#    Acw[0,1] = Acw[2,3] = math.sin(w*Ts)/w
#    Acw[0,3] =  (1-math.cos(w*Ts))/w
#    Acw[2,1] = - Acw[0,3]
#    Acw[1,1] = Acw[3,3] = math.cos(w*Ts)
#    Acw[1,3] = -math.sin(w*Ts)
#    Acw[3,1] = -Acw[1,3]
#    
#        #coordinated turn ccw
#    w = 10*math.pi/180.0;
#    Accw = np.identity(4)
#    Accw[0,1] = Accw[2,3] = math.sin(w*Ts)/w
#    Accw[0,3] =  (1-math.cos(w*Ts))/w
#    Accw[2,1] = - Accw[0,3]
#    Accw[1,1] = Accw[3,3] = math.cos(w*Ts)
#    Accw[1,3] = -math.sin(w*Ts)
#    Accw[3,1] = -Accw[1,3]
#    
#    #Starts at the origin with nonzero velocity
#    X = np.array([0,0,0,2])
#
#    data = MMData()
#
#    samplesPerTurn = int(math.ceil(3*math.pi/(2*w*Ts)))
#    samplesPerStraight = 29
#    samplesPerLoop = 2*samplesPerTurn+4*samplesPerStraight
#    Nsamples = Nloops*samplesPerLoop
#
#    #State vector samples
#    data.x = np.zeros([Nsamples,4])
#    data.x[0,:] = X
#    
#    #Model samples
#    data.num_models = 3
#    data.m = np.array([0]*(Nsamples))
#    
#    data.Qm = [(sigmaLik**2)*np.identity(4),(sigmaLik**2)*np.identity(4),(sigmaLik**2)*np.identity(4)]
#    data.R =  [(sigma**2)*np.identity(4),(sigma**2)*np.identity(4),(sigma**2)*np.identity(4)]
#
#    for l in range(0,Nloops,1):                         
#        for i in range(1,samplesPerStraight,1):
#            A = Acv
#            m = 0
#            X = np.dot(A,X) 
#            data.x[i+(l)*samplesPerLoop,:] = X #+np.random.randn(4)*sigma
#            data.m[i+(l)*samplesPerLoop] = m
#               
#
#        for i in range(samplesPerStraight,samplesPerStraight+samplesPerTurn,1):
#            A = Accw
#            m = 1
#            X = np.dot(A,X) 
#            data.x[i+(l)*samplesPerLoop,:] = X #+np.random.randn(4)*sigma
#            data.m[i+(l)*samplesPerLoop] = m
#
#        for i in range(samplesPerStraight+samplesPerTurn,3*samplesPerStraight+samplesPerTurn,1):
#            A = Acv
#            m = 0
#            X = np.dot(A,X) 
#            data.x[i+(l)*samplesPerLoop,:] = X #+np.random.randn(4)*sigma
#            data.m[i+(l)*samplesPerLoop] = m
#        
#        for i in range(3*samplesPerStraight+samplesPerTurn,3*samplesPerStraight+2*samplesPerTurn,1):
#            A = Acw
#            m = 2
#            X = np.dot(A,X) 
#            data.x[i+(l)*samplesPerLoop,:] = X #+np.random.randn(4)*sigma
#            data.m[i+(l)*samplesPerLoop] = m
#         
#        for i in range(3*samplesPerStraight+2*samplesPerTurn,4*samplesPerStraight+2*samplesPerTurn,1):
#            A = Acv
#            m = 0
#            X = np.dot(A,X) 
#            data.x[i+(l)*samplesPerLoop,:] = X #+np.random.randn(4)*sigma
#            data.m[i+(l)*samplesPerLoop] = m
#        
#    noisy_data = copy.deepcopy(data)
#    noisy_data.x += np.random.randn(*noisy_data.x.shape)*sigma
#
#    return data, noisy_data
#    
def jet_fly_random(Nactions,Ts, sigma, sigmaLik):
    
    #X = [x1 x1' x2 x2'] aircraft position in arbitrary coord. frame
    
    #Constant Velocity (CV)
    Acv = np.identity(4)
    Acv[0,1] = Acv[2,3] = Ts

    #Coordinated turn ccw
    w = -5*math.pi/180.0;
    Acw = np.identity(4)
    Acw[0,1] = Acw[2,3] = math.sin(w*Ts)/w
    Acw[0,3] =  (1-math.cos(w*Ts))/w
    Acw[2,1] = - Acw[0,3]
    Acw[1,1] = Acw[3,3] = math.cos(w*Ts)
    Acw[1,3] = -math.sin(w*Ts)
    Acw[3,1] = -Acw[1,3]
    
    #coordinated turn ccw
    w = 5*math.pi/180.0;
    Accw = np.identity(4)
    Accw[0,1] = Accw[2,3] = math.sin(w*Ts)/w
    Accw[0,3] =  (1-math.cos(w*Ts))/w
    Accw[2,1] = - Accw[0,3]
    Accw[1,1] = Accw[3,3] = math.cos(w*Ts)
    Accw[1,3] = -math.sin(w*Ts)
    Accw[3,1] = -Accw[1,3]

    print "Model CV: A is \n"+str(Acv)
    print "Model CCW: A is \n"+str(Accw)
    print "Model CW: A is \n"+str(Acw)


    #Starts at the origin with nonzero velocity
    X = np.array([0,0,0,2])

    data = MMData()

    samplesPerAction = 50
    Nsamples = Nactions*samplesPerAction + 150

    #State vector samples
    data.x = np.zeros([Nsamples,4])
    data.x[0,:] = X
    
    #Model samples
    data.num_models = 3
    data.m = np.array([0]*(Nsamples))
    
    sigma = 0.1 #Process noise
    data.Qm = [(sigmaLik**2)*np.identity(4),(sigmaLik**2)*np.identity(4),(sigmaLik**2)*np.identity(4)]
    data.R = [(sigma**2)*np.identity(4),(sigma**2)*np.identity(4),(sigma**2)*np.identity(4)]
    
        #Start with straight then CCW then CW to make sure all are present
    for i in range(0,50,1):
        m = 0
        A = Acv
        X = np.dot(A,X)
        data.x[i,:] = X
        data.m[i] = m

    for i in range(50,100,1):
        m = 1
        A = Accw
        X = np.dot(A,X)
        data.x[i,:] = X
        data.m[i] = m
         
    for i in range(100,150,1):
        m = 2
        A = Acw
        X = np.dot(A,X)
        data.x[i,:] = X
        data.m[i] = m
          
    for a in range(0,Nactions,1):                           
        m = random.randint(0,2)
        if m == 0:
            A = Acv
        elif m == 1:
            A = Accw
        else:
            A = Acw

        for i in range(0, samplesPerAction, 1):
            X = np.dot(A,X)
            data.x[i+150+(a)*samplesPerAction,:] = X
            data.m[i+150+(a)*samplesPerAction] = m  

    #pdb.set_trace()

    noisy_data = copy.deepcopy(data)
    noisy_data.x += np.random.randn(*noisy_data.x.shape)*sigma

    return data, noisy_data

def jet_fly_short_random(Nactions,Ts, sigma, sigmaLik):
    
    #X = [x1 x1' x2 x2'] aircraft position in arbitrary coord. frame
    
    #Constant Velocity (CV)
    Acv = np.identity(4)
    Acv[0,1] = Acv[2,3] = Ts

    #Coordinated turn ccw
    w = -5*math.pi/180.0;
    Acw = np.identity(4)
    Acw[0,1] = Acw[2,3] = math.sin(w*Ts)/w
    Acw[0,3] =  (1-math.cos(w*Ts))/w
    Acw[2,1] = - Acw[0,3]
    Acw[1,1] = Acw[3,3] = math.cos(w*Ts)
    Acw[1,3] = -math.sin(w*Ts)
    Acw[3,1] = -Acw[1,3]
    
    #coordinated turn ccw
    w = 5*math.pi/180.0;
    Accw = np.identity(4)
    Accw[0,1] = Accw[2,3] = math.sin(w*Ts)/w
    Accw[0,3] =  (1-math.cos(w*Ts))/w
    Accw[2,1] = - Accw[0,3]
    Accw[1,1] = Accw[3,3] = math.cos(w*Ts)
    Accw[1,3] = -math.sin(w*Ts)
    Accw[3,1] = -Accw[1,3]

    print "Model CV: A is \n"+str(Acv)
    print "Model CCW: A is \n"+str(Accw)
    print "Model CW: A is \n"+str(Acw)


    #Starts at the origin with nonzero velocity
    X = np.array([0,0,0,2])

    data = MMData()

    samplesPerAction = 3
    Nsamples = Nactions*samplesPerAction + 150

    #State vector samples
    data.x = np.zeros([Nsamples,4])
    data.x[0,:] = X
    
    #Model samples
    data.num_models = 3
    data.m = np.array([0]*(Nsamples))
    
    sigma = 0.1 #Process noise
    data.Qm = [(sigmaLik**2)*np.identity(4),(sigmaLik**2)*np.identity(4),(sigmaLik**2)*np.identity(4)]
    data.R = [(sigma**2)*np.identity(4),(sigma**2)*np.identity(4),(sigma**2)*np.identity(4)]
    
        #Start with straight then CCW then CW to make sure all are present
    for i in range(0,50,1):
        m = 0
        A = Acv
        X = np.dot(A,X)
        data.x[i,:] = X
        data.m[i] = m

    for i in range(50,100,1):
        m = 1
        A = Accw
        X = np.dot(A,X)
        data.x[i,:] = X
        data.m[i] = m
         
    for i in range(100,150,1):
        m = 2
        A = Acw
        X = np.dot(A,X)
        data.x[i,:] = X
        data.m[i] = m
          
    for a in range(0,Nactions,1):                           
        m = random.randint(0,2)
        if m == 0:
            A = Acv
        elif m == 1:
            A = Accw
        else:
            A = Acw

        for i in range(0, samplesPerAction, 1):
            X = np.dot(A,X)
            data.x[i+150+(a)*samplesPerAction,:] = X
            data.m[i+150+(a)*samplesPerAction] = m  

    #pdb.set_trace()

    noisy_data = copy.deepcopy(data)
    noisy_data.x += np.random.randn(*noisy_data.x.shape)*sigma

    return data, noisy_data


def jet_fly_lawnmower(n_loops = 5, ymin = 0.0, ymax = 20.0, v0 = 2.0, Ts = 0.2, w = 5.0*math.pi/180.0, sigmaLik = 0.005, sigma = 0.05):
    
    #X = [x1 x1' x2 x2'] aircraft position in arbitrary coord. frame
    
    #Constant Velocity (CV)
    Acv = np.identity(4)
    Acv[0,1] = Acv[2,3] = Ts

    #Coordinated turn ccw
    w = -w;
    Acw = np.identity(4)
    Acw[0,1] = Acw[2,3] = math.sin(w*Ts)/w
    Acw[0,3] =  (1-math.cos(w*Ts))/w
    Acw[2,1] = - Acw[0,3]
    Acw[1,1] = Acw[3,3] = math.cos(w*Ts)
    Acw[1,3] = -math.sin(w*Ts)
    Acw[3,1] = -Acw[1,3]
    
    #coordinated turn ccw
    w = -w;
    Accw = np.identity(4)
    Accw[0,1] = Accw[2,3] = math.sin(w*Ts)/w
    Accw[0,3] =  (1-math.cos(w*Ts))/w
    Accw[2,1] = - Accw[0,3]
    Accw[1,1] = Accw[3,3] = math.cos(w*Ts)
    Accw[1,3] = -math.sin(w*Ts)
    Accw[3,1] = -Accw[1,3]

    print "Model CV: A is \n"+str(Acv)
    print "Model CCW: A is \n"+str(Accw)
    print "Model CW: A is \n"+str(Acw)


    #Starts at the origin with nonzero velocity
    X = np.array([0,0,ymin,v0])

    data = MMData()

    #State vector samples
    data.x = X
    
    #Model samples
    data.num_models = 3
    m = 0
    lm = 'up'
    A = Acv

    data.m = np.array([m])
    
    data.Qm = [(sigmaLik**2)*np.identity(4),(sigmaLik**2)*np.identity(4),(sigmaLik**2)*np.identity(4)]

    loop_cnt = 0       
    delta = v0*0.001
    while loop_cnt < n_loops:
        X = np.dot(A, X)
        data.x = np.vstack((data.x, X))

        if(X[2] >= ymax and lm == 'up'):
            lm = 'cw'
            A = Acw
            m = 1
        if(X[3] <= -(v0-delta) and lm == 'cw'):
            lm = 'down'
            A = Acv
            m = 0
        if(X[2] <= ymin and lm == 'down'):
            lm = 'ccw'
            A = Accw
            m = 2
        if(X[3] >= v0-delta and lm == 'ccw'):
            lm = 'up'
            A = Acv
            m = 0
            loop_cnt += 1    

        data.m = np.hstack((data.m, m))

    noisy_data = copy.deepcopy(data)
    noisy_data.x += np.random.randn(*noisy_data.x.shape)*sigma

    return data, noisy_data

#   Simulates a very simplified temperature control system: if the 
#   temperatures rises above a threshold, it turns the AC on and then
#   the temperature starts dropping. Once the temperature drops below
#   another threshold, the AC is turned off and the temperature starts
#   to rise again
#def bang_bang_ac(Nsamples,Ts,prob=-1.0):
        
#    Ah = np.identity(2)
#    Ah[0,1]=0.3*Ts #temperature increment
    
    #Colder
#    Ac = np.identity(2)
#    Ac[0,1]=-0.2*Ts #temperature decrement
        
#    data = MMData()
    
    #Starts at 70F
#    X = np.array([70.0,1.0])
#    A = Ah  #Starts with the AC off
#    m=0
    
    #State vector samples
#    data.x = np.zeros([Nsamples,2])
#    data.x[0,:] = X
    
    #Model samples
#    data.num_models = 2
#    data.m = np.array([0]*(Nsamples))
#    data.m[0] = m #Starts with the AC off
    
#    sigma = 0.1 #Process noise
#    data.Qm = [np.array([sigma**2]),np.array([sigma**2])]
    
#    for i in range(1,Nsamples,1):                       
#        X = np.dot(A,X) 
#        data.x[i,:] = X+np.array([1.0,0.0])*np.random.randn(1)*sigma
        
#        if X[0] > 73.0:         #Too hot            
#            if(prob>0.0):               
#                #Transitions with probability prob
#                if(np.random.rand(1)<prob):
#                    A = Ac #Turns AC on
#                    m = 1               
#            else:           
#                A = Ac #Turns AC on
#                m = 1
#        
#        elif (X[0] < 70.0):     #Too cold           
#            if(prob>0.0):
#                #Transitions with probability prob
#                if(np.random.rand(1)<prob):
#                    A = Ah  #Turns AC off
#                    m = 0               
#            else:
#                A = Ah  #Turns AC off
#                m = 0
                
#        data.m[i] = m
        
    
#    return data

#Simulates a switched RC circuit that charges through one load
#and discharges through a different one depending on voltages 
#thresholds
def rc_circuit(Nsamples,Ts,Rl,Rc,C,X0=5.0,Vin=10.0,Vmin=3.0,Vmax=4.0, sigma=0.1, sigmaLik =0.01, prob=-1.0):
    
    Ac = np.row_stack([(Rc*C)/(Rc*C+Ts)]) #charge dynamics
    Bc = np.row_stack([Ts/(Rc*C+Ts)])
    Ad = np.row_stack([(Rl*C)/(Rl*C+Ts)]) #discharge dynamics
    Bd = np.row_stack([0.0])

    print "A0 = " + str(Ac)
    print "B0 = " + str(Bc)
    print "Ad = " + str(Ad)
    print "Bd = " + str(Bd)
            
    data = MMData()
    noisy_data = MMData()
        
    X = np.row_stack([X0])  #Initial voltage    
    m = 0 #Starts charging  
        
    #State vector samples
    data.x = np.zeros([Nsamples,1])
    data.x[0,:] = X 
    data.u = np.row_stack(np.ones([Nsamples])*Vin)
    
    noisy_data.x = np.zeros([Nsamples,1])
    noisy_data.x[0,:] = X
    noisy_data.u = np.row_stack(np.ones([Nsamples])*Vin)

    #Model samples
    data.num_models = 2
    data.m = np.array([0]*(Nsamples))
    data.m[0] = m #Starts with the AC off

    noisy_data.num_models = 2
    noisy_data.m = np.array([0]*(Nsamples))
    noisy_data.m[0] = m #Starts with the AC off



    #Noise covariance matrices      
    data.Qm = [np.array([sigmaLik**2]),np.array([sigmaLik**2])]
    data.R = [np.array([sigma**2]),np.array([sigma**2])]    
    for i in range(1,Nsamples,1):                       
        
        if m == 0:
            X = np.dot(Ac,X)+np.dot(Bc,data.u[i-1])
        else:
            X = np.dot(Ad,X)+np.dot(Bd,data.u[i-1])
                
        data.x[i] = X
                #noisy_data.x[i] = X+np.random.randn(1)*sigma #adds noise
        
        if X > Vmax: # starts discharging
            if((prob>0.0 and np.random.rand(1)<prob) or prob<0.0):              
                m = 1                       
        elif (X < Vmin):    # starts charging
            if((prob>0.0 and np.random.rand(1)<prob) or prob<0.0):              
                m = 0                               
        
        data.m[i] = m       
        noisy_data.m[i] = m
    
    noisy_data = copy.deepcopy(data)
    noisy_data.x += np.random.randn(*noisy_data.x.shape)*sigma

    return data, noisy_data
#Reperesents a set of RC Circuits in series
#To simplify the code, each model is assumed to have the same parameters
#Instead of generating each possible matrix, the current state matrix is calculated from a boolean vector indicating whether that circuit is charging or discharging
#The model number for a particular combination of charging and discharing circuits is set by turning the boolean vector (taken as a binary number) into an integer index
#def rc_circuit_series(Ncircuits, Nsamples,Ts,Rl,Rc,C,X0=5.0,Vin=10.0,Vmin=3.0,Vmax=4.0, sigma=0.005,sigmaLik = 0.005, prob=-1.0):   
#    A = np.zeros((Ncircuits, Ncircuits))#charge dynamics
#    B = np.zeros((Ncircuits, Ncircuits))
#        
#    data = MMData()
#        
#    X = np.random.uniform(Vmin, Vmax, Ncircuits)    #Initial voltage    
#    chargeState = np.random.binomial(1,0.5,Ncircuits)
#        m = int( (len(chargeState) * '%d') % tuple(chargeState) ,2)
#
#
#    #State vector samples
#    data.x = np.zeros([Nsamples,Ncircuits])
#    data.x[0,:] = X 
#    data.u = np.zeros((Nsamples, Ncircuits))
#        data.u[:,0] = Vin
#
#    #Model samples
#    data.num_models = 2**Nsamples
#    data.m = np.array([0]*(Nsamples))
#    data.m[0] = m #Starts with the AC off
#        #pdb.set_trace()
#
#    #Cosnstruct covariance matrices     
#    data.Qm = [np.array([sigmaLik**2]),np.array([sigmaLik**2])]
#        noisy_data.Qm = [np.array([sigmaLik**2]),np.array([sigmaLik**2])]
#    data.R = [np.array([sigma**2]),np.array([sigma**2])]    
#
#        Ac = (Rc*C)/(Rc*C+Ts) #charge dynamics
#    Bc = Ts/(Rc*C+Ts)
#    Ad = (Rl*C)/(Rl*C+Ts) #discharge dynamics
#    Bd = 0.0
#
#    #Construct Initial A and B
#        if chargeState[0]:
#            A[0,0] = Ac
#            B[0, 0] = Bc
#        else: 
#            A[0, 0] = Ad
#            B[0, 0] = Bd
#
#        for j in range(1, Ncircuits):
#            if chargeState[j]:
#                A[j,j] = Ac
#                A[j, j-1] = Bc 
#            else:
#                A[j] = Ad
#                A[j, j-1] = Bd
#    
#        for i in range(1,Nsamples,1):                       
#        #Calculate new X
#        #pdb.set_trace()
#                X = np.dot(A,X)+np.dot(B,data.u[i-1])
#                #Add X to data vector                       
#        data.x[i] = X+np.random.randn(1)*sigma #adds noise
#        
#                #Update charging and discharging states
#                for j in range(0,Ncircuits):
#                    if ((X[j] >= Vmax) and ((prob>0.0 and np.random.rand(1)<prob) or prob<0.0)):        
#                chargeState[j] = 0
#                    elif ((X[j] <= Vmin) and ((prob>0.0 and np.random.rand(1)<prob) or prob<0.0)):
#                        chargeState[j] = 1 
#        
#                #Update model index
#            m = int( (len(chargeState) * '%d') % tuple(chargeState) ,2)
#                data.m[i] = m       
#
#                
#                
#                #Update A and B
#                if chargeState[0]:
#                    A[0,0] = Ac
#                    B[0, 0] = Bc
#                else: 
#                    A[0, 0] = Ad
#                    B[0, 0] = Bd
#
#                for j in range(1, Ncircuits):
#                    if chargeState[j]:
#                        A[j,j] = Ac
#                        A[j, j-1] = Bc 
#                    else:
#                        A[j] = Ad
#                        A[j, j-1] = Bd
#    return data
#

