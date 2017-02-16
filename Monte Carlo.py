import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import fmin, fmin_powell,fmin_bfgs,fmin_cg
from scipy import mean
import pandas as pd
from Binomial_Tree import find_binomial_option_prices

def generate_paths(num_paths,time_steps, time_horizon, rf, volitality, S0):
    # generate n risk-neutral paths for the stock movement
    
    M = num_paths 	# Number of paths
    N = time_steps	 # Number of time steps
    T = time_horizon 	# Number of paths
        
    dt = T*1.0/N	# simulation time step    
    S = np.zeros((M,N+1))
    S[:,0] = S0 # Set the starting price as S0
    
    #Generating paths
    Wt = np.random.normal(0, 1, (M,N))
    S[:,1:] = np.exp((rf-0.5*volitality**2)*dt + Wt*volitality*np.sqrt(dt))
    S = np.cumprod(S, axis = 1)
 
    return S   

def find_total_simulation_valule(critical_values, S, K, T, N):
    
    value = S[:,1:-1].copy()
    dt = T*1.0/N
#    critical_values = [100] *11

#    critical_values = critical_values + [K]
#     critical_values = critical_values. 

    for i in range (N):
        value[:,i:i+1] =np.where((value[:,i:i+1]< critical_values[i]) & (value[:,i:i+1]<> 0), K - value[:,i:i+1],0)
        for j in list(np.where((value[:,i:i+1]< critical_values[i]) & (value[:,i:i+1]<> 0)))[0]:
            value[j:j+1,i+1:] = 0
    
            if (i == N-1) & (value[j:j+1,:-1].all == 0):
                value[j,-1] = np.maximum(K - S[j,-1], 0)
    

    cumulative_dt = dt*(np.cumsum(np.ones(value.shape), axis =1)-1)
    discount_factor = np.exp(-rf*cumulative_dt)
    discounted_value =  discount_factor * value       
    path_value = np.sum(discounted_value,axis = 1)
    option_value = - np.mean(path_value)
    
    return option_value

def plot_single_boundary(opt_bound):

    pl.figure(figsize=(10,6))
    pl.axis([0, 12, 50, 120])
    pl.title('Decision Boudaries')
    pl.xlabel('time steps')
    pl.ylabel('asset price')
    pl.plot(opt_bound)
    pl.show()


    return 0

def plot_paths(S):
    pl.figure(figsize=(10,6))
    for i in range (0,50):
         pl.plot(np.linspace(0,12,13), S[i])
    pl.title('Geometric Brownian Motion prices')
    pl.xlabel('time steps')
    pl.ylabel('asset prices')
    
    pl.show() 
    return 0

def find_decision_points_fmin(x0,S,K):
    print "Optimization start"
    critical_values = fmin(find_total_simulation_valule, x0, args=(S,K, T, N))
    critical_values[critical_values > K] = K
#    critical_values = critical_values[-1] 
    option_value = -find_total_simulation_valule(critical_values, S, K, T, N)                  
       
    return [critical_values,option_value]

def get_multiple_runs(num_paths,time_steps, time_horizon, rf, volitality, S0, num_runs, x0, K):
    
    bound_list = []
    option_values = []
    for i in range(num_runs):
        print 'Simulating run' + ' ' + str(i+1)    
        S = generate_paths(num_paths,time_steps, time_horizon, rf, volitality, S0)
        plot_paths(S)
        opt_bound, opt_value = find_decision_points_fmin(x0,S,K)
        option_values.append(opt_value)
        plot_single_boundary(opt_bound)
        bound_list.append(opt_bound)
        boundaries = pd.DataFrame.from_records(bound_list,columns = range(1,time_steps+1))
        option_value =  mean(option_values)
    return [boundaries,option_value]


def find_boundary_interval(boundaries):
    average_boundary = boundaries.mean(axis = 0)
    std_boundary = boundaries.std(axis = 0)
    upper_95_interval = average_boundary + 1.96*std_boundary
    lower_95_interval = average_boundary - 1.96*std_boundary
    
    return [average_boundary,upper_95_interval,lower_95_interval]
    
def plot_boundary(geomertic_boundaries, binomial_boundaries,upper_boundaries, lower_boundaries, num_runs, num_paths):
    
    
    pl.figure(figsize=(10,6))
    pl.axis([0, 12, 50, 120])
    pl.title('Decision Boudaries -' + str(num_runs) + ' runs , ' + str(num_paths) + ' paths')
    pl.xlabel('time steps')
    pl.ylabel('asset price')
    pl.plot(geomertic_boundaries, label = 'Monte Carlo Decision Boundary')
    pl.plot(upper_boundaries, label = '95 % Confidence Upper Bound')
    pl.plot(lower_boundaries, label = '95 % Confidence Lower Bound')
    pl.plot(binomial_boundaries, label = 'Binomial Tree Decision Boundary' )
    pl.legend(loc = 'bottom right')

    figure = pl.gcf()
    pl.show()
    figure.savefig(\
                   'C:\Users\meloc\Desktop\University\Courses\Thesis\Graphs\\' \
                   + str(num_runs) + ' runs - ' + str(num_paths) + ' paths.png')
    
    return 0

if __name__ == "__main__":  
    
    rf = 0.03
    volitality =  0.15
    num_paths = 100000
    N = 12 #time steps
    time_steps = N
    T = 1.0 #time horizon
    time_horizon = T
    S0 = 100
  
    K = 100
    num_runs = 50

    x0 = [100]*12
    
    geometric_boundaries, gemetrc_option_value = get_multiple_runs(num_paths,N, T, rf, volitality, S0, num_runs, x0, K)
    average_geometric_boundaries, upper_95_boundary, lower_95_boundary= find_boundary_interval(geometric_boundaries)
    average_geometric_boundaries = average_geometric_boundaries[:-1]
    upper_95_boundary = upper_95_boundary[:-1]
    lower_95_boundary = lower_95_boundary[:-1]

    binomial_option_value, binomial_boundaries = find_binomial_option_prices(1000, T, rf, volitality, S0, K, -1, am = False)
    binomial_boundaries = binomial_boundaries[:-1]
    plot_boundary(average_geometric_boundaries, binomial_boundaries, upper_95_boundary, lower_95_boundary, num_runs, num_paths)
    print "The option value of using Monte Carlo Simulation is : ", gemetrc_option_value
    print "The option value of using binomial tree is : ", binomial_option_value
