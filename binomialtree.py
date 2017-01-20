
import math
import numpy as np
import matplotlib.pyplot as pl

class stock_options:
    
    def __init__(self,S0,K,r,T, N,parameters):
        # the initiation of the master class object, with the parameters of S0
        # K, r, T
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T

        self.tree_node = None  
        # the parameters will be inputed as a dictionary, each parameter will be
        # recorded as an object of the stock_options class
        #self.mu = parameters.get("mu", 0)  
        self.sigma = parameters.get("sigma", 0)  
        self.is_call = parameters.get("is_call", True)  
        self.is_european = parameters.get("is_eu", True)  
        self.N = N
        self.dt = T/float(N) 
        self.discount = math.exp(-r*self.dt)
        
        
class BinomialTreeOption(stock_options):
    # inherented fromm master class stock_options
    
    
    def _initialize_risk_neutral_parameters_(self):
        self.u = math.exp(self.sigma*math.sqrt(self.dt))  
        self.d = math.exp(-self.sigma*math.sqrt(self.dt))

        self.qu = (math.exp((self.r)*self.dt) - self.d)/(self.u-self.d)
        self.qd = 1-self.qu
         
    def _init_binomial_tree(self):
        # initialize the tree nodes for N periods
        self.tree_node = [np.array([self.S0])]
                     
        for i in range(self.N):
            previous_branches = self.tree_node[-1]
            new_node = np.concatenate((previous_branches*self.u, [previous_branches[-1]*self.d]))
                                 
            self.tree_node.append(new_node) 
     
    def _initialize_payoff_tree_(self):
        
        return np.maximum(0, (self.tree_node[self.N]-self.K) if self.is_call   
                          else (self.K-self.tree_node[self.N]))
    
    def _check_early_exercise(self, holding_payoff, node):
        
        exercise_payoff = self.K - self.tree_node[node]
                        
        return np.maximum(exercise_payoff, holding_payoff)
        
    def _get_early_exercise_boundary(self, holding_payoff, node):
        
        stock_price = self.tree_node[node+1]
        exercise_payoff = self.K - self.tree_node[node]
       

        early_exercise = []
        for i in reversed(range(len(holding_payoff))):
            if exercise_payoff[i] >= holding_payoff[i]:
                #print "early exercised at  %s" % stock_price[i]
                early_exercise.append(stock_price[i])
                #print early_exercise
            else:
                early_exercise.append(None)

     
        exercise_boundary = max(early_exercise)
        #exercise_boundary = np.maximum(exercise_values)
        
        
        return exercise_boundary
        
        
    def _traverse_tree(self, payoff):
        exercise_boundaries = []
        for i in reversed(range(self.N)):
            payoff = (payoff[:-1] * self.qu + payoff[1:] * self.qd) * self.discount

            if not self.is_european:
                exercise_boundary = self._get_early_exercise_boundary(payoff, i) 
                payoff = self._check_early_exercise(payoff, i) 
                
            exercise_boundaries =[exercise_boundary] + exercise_boundaries
        
        #exercise_boundaries = np.asarray(exercise_boundaries)        
   
        return [payoff,exercise_boundaries]
    

    def values(self):
        self._initialize_risk_neutral_parameters_()
        self._init_binomial_tree()
        values = self._traverse_tree(self._initialize_payoff_tree_())

        return values   
    

if __name__ == "__main__":
    
    #S0,K,r,T, N,parameters
    american_option = BinomialTreeOption(100, 100, 0.03, 1, 12,
        {"sigma": 0.15,"is_call": False, "is_eu": False})
    Ame_price = american_option.values()[0][0]
    exercise_boundaries =  [None] + american_option.values()[1]
    A = american_option.tree_node
    pl.figure(figsize=(10,6))
    pl.axis([0, 12, 40, 120])
    pl.title('exercise boundary - binomial tree, S0 = 100, K = 100')
    pl.xlabel('t')
    pl.ylabel('asset price')
    print exercise_boundaries
    pl.plot(exercise_boundaries)
   

    
        
        
        
    