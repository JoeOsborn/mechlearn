import testing
import pdb
import pickle

def rc_circuit_em(nReps):
    output = open('rc_circuit_data.pkl','w')
    
    for i in range(nReps):
        success = False
        result = None
        while not(success):
            try:
                result = testing.test_em(1000)
                success = 1                
                break
            except ValueError:
                success = False
        pickle.dump(result, output)
    output.close()
    return 1

def rc_circuit_filter():
    
    return 1

def rc_circuit_jmls():
    
    return 1

def rc_circuit_jmls_filter():
    return 1

def rc_circuit_series_em(nReps, nCircuits):
     output = open('rc_series_data.pkl','w')
    
    for i in range(nReps):
        success = False
        result = None
        while not(success):
            try:
                result = testing.test_em(1000)
                success = 1                
                break
            except ValueError:
                success = False
        pickle.dump(result, output)
    output.close()
    return 1

def rc_series_filter():
    return 1

def rc_series_jmls():
    return 1

def rc_series_jmls_filter():
    return 1

def fly_racetrack_em(nReps, nLoops):
    output = open('fly_racetrack_data.pkl','w')
    
    for i in range(nReps):
        success = False
        result = None
        while not(success):
            try:
                result = testing.test_racetrack_em(nLoops)
                success = 1                
                break
            except ValueError:
                success = False
        pickle.dump(result, output)
    output.close()
    return 1

def fly_racetrack_filter():
    return 1

def fly_racetrack_jmls():
    return 1

def fly_racetrack_jmls_filter():
    return 1

def fly_random_em(nReps, nActions):
    output = open('fly_random_data.pkl','w')
    
    for i in range(nReps):
        success = False
        result = None
        while not(success):
            try:
                result = testing.test_flight_em(nActions)
                success = 1                
                break
            except ValueError:
                success = False
        pickle.dump(result, output)
    output.close()
    return 1

def fly_random_em():
    return 1

def fly_random_jmls():
    return 1

def fly_random_jmls_filter():
    return 1
