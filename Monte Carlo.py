import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import fmin
import pandas as pd
from binomialtree import BinomialTreeOption

def generate_paths(num_paths,time_steps, time_horizon, rf, volitality, S0):
    # generate n risk-neutral paths for the stock movement
    
    M = num_paths 	# Number of paths
    N = time_steps	 # Number of time steps
    T = time_horizon 	# Number of paths
        
    dt = T/N	# simulation time step    
    S = np.zeros((M,N+1))
    S[:,0] = S0 # Set the starting price as S0
    
    #Generating paths
    Wt = np.random.normal(0, 1, (M,N))
    S[:,1:] = np.exp((rf-0.5*volitality**2)*dt + Wt*volitality*np.sqrt(dt))
    S = np.cumprod(S, axis = 1)
 
    return S   

def find_total_simulation_valule(critical_values, S, K, T, N):
    
    value = S[:,1:].copy()

    dt = T*1.0/N
#    critical_values = [100]*12
    for i in range (N):
        value[:,i:i+1] =np.where((value[:,i:i+1]< critical_values[i]) & (value[:,i:i+1]<> 0), K - value[:,i:i+1],0)
        for j in list(np.where((value[:,i:i+1]< critical_values[i]) & (value[:,i:i+1]<> 0)))[0]:
            value[j:j+1,i+1:] = 0
    value[:,-1] = np.maximum(K - S[:,-1], 0)

    cumulative_dt = dt*(np.cumsum(np.ones(value.shape), axis =1)-1)
    discount_factor = np.exp(-rf*cumulative_dt)
    discounted_value =  discount_factor * value       
    path_value = np.sum(discounted_value,axis = 1)
    option_value = - np.mean(path_value)
    
    return option_value

def plot_paths(S):
    for i in range (0,50):
         pl.plot(np.linspace(0,12,13), S[i])
    pl.title('asset prices')
    pl.xlabel('time steps')
    pl.figure(figsize=(10,6))
    pl.show() 
    return 0

def find_decision_points_fmin(x0,S,K):
    xopt = fmin(find_total_simulation_valule, x0, xtol=1e-9, args=(S,K, T, N))
    xopt[xopt>K] = K
    
    
    return xopt

def get_multiple_runs(num_paths,time_steps, time_horizon, rf, volitality, S0, num_runs, x0, K):
    
    bound_list = []
    for i in range(num_runs):
        print 'Simulating run' + ' ' + str(i+1)    
        S = generate_paths(num_paths,time_steps, time_horizon, rf, volitality, S0)
        plot_paths(S)
        opt_bound = find_decision_points_fmin(x0,S,K)
        bound_list.append(opt_bound)
        boundaries = pd.DataFrame.from_records(bound_list,columns = range(1,time_steps+1))
   
    return boundaries

#def plot_binomial_boundary():
#    american_option = BinomialTreeOption(100, 100, 0.03, 1, 12,
#    {"mu": 0.03, "sigma": 0.15, "is_call": False, "is_eu": False})
#    Ame_price = american_option.values()[0][0]
#    exercise_boundaries =  american_option.values()[1]
#
#    pl.figure(figsize=(10,6))
#    pl.axis([0, 12, 40, 100])
#    pl.title('exercise boundary - binomial tree, S0 = 100, K = 100')
#    pl.xlabel('t')
#    pl.ylabel('asset price')
#   # exercise_boundaries = exercise_boundaries[exercise_boundaries>1]
#    pl.plot(exercise_boundaries)
#    
#    return 0

def find_boundary_interval(boundaries):
    average_boundary = boundaries.mean(axis = 0)
    std_boundary = boundaries.std(axis = 0)
    upper_95_interval = average_boundary + 1.96*std_boundary
    lower_95_interval = average_boundary - 1.96*std_boundary
    
    return [average_boundary,upper_95_interval,lower_95_interval]
    
def plot_boundary(geomertic_boundaries, binomial_boundaries,upper_boundaries, lower_boundaries):
    
    
    pl.figure(figsize=(10,6))
    pl.axis([0, 12, 50, 120])
    pl.title('Decision Boudaries')
    pl.xlabel('time steps')
    pl.ylabel('asset price')
    pl.plot(geomertic_boundaries)
    pl.plot(upper_boundaries)
    pl.plot(lower_boundaries)
    pl.plot(binomial_boundaries)
    pl.show
    
    return 0

if __name__ == "__main__":  
    
    rf = 0.03
    volitality =  0.15
    num_paths = 50000
    N = 12 #time steps
    T = 1.0 #time horizon
    S0 = 100
  
    K = 100
    num_runs = 100

    x0 = [100]*12
    K = 100

    geometric_boundaries = get_multiple_runs(num_paths,N, T, rf, volitality, S0, num_runs, x0, K)
    average_geometric_boundaries = find_boundary_interval(geometric_boundaries)[0]
    upper_95_boundary = find_boundary_interval(geometric_boundaries)[1]
    lower_95_boundary = find_boundary_interval(geometric_boundaries)[2]
    american_option = BinomialTreeOption(100, 100, 0.03, 1, 12, {"sigma": 0.15,"is_call": False, "is_eu": False})
    Ame_price = american_option.values()[0][0]
    binomial_boundaries = [None] + american_option.values()[1]
    plot_boundary(average_geometric_boundaries, binomial_boundaries, upper_95_boundary, lower_95_boundary)
    
#    plot_binomial_boundary()
#    plot_geometric_boundary(boundaries)   


