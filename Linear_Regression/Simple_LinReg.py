import numpy as np
import matplotlib.pyplot as plt
import time

class Simple_LinearRegression:
    '''
    Algorithm is centered around a single feature corresponding to a single label.
    Plots can be made without a linear combination of features

    Weight and bias is updated iteratively with a gradient descent equation
    Cost is iteratively collected and can be plotted... a converging plot verifies the algorithm
    '''

    def __init__(self, X, y, rate):
        self.x_data = X
        self.y_data = y
        
        self.weight = 0
        self.bias = 0
        self.learning_rate = rate
        
        self.total_samples = self.x_data.size
        
        self.cost_history = []
        self.iter_history = []
        
        
    def cost_function(self):
        error = 0.0
        
        for i in range(self.total_samples):
            error += (self.y_data[i] - (self.weight * self.x_data[i] + self.bias)) ** 2
        
        return error / self.total_samples
    
    
    def update_weights(self):
        weight_der = 0
        bias_der = 0
        
        for i in range(0, self.total_samples):
            weight_der += -2 * self.x_data[i] * (self.y_data[i] - ((self.weight * self.x_data[i]) + self.bias))
            bias_der += -2 * (self.y_data[i] - ((self.weight * self.x_data[i]) + self.bias))
            
        self.weight -= (weight_der / float(self.total_samples)) * self.learning_rate
        self.bias -= (bias_der / float(self.total_samples)) * self.learning_rate
        
        return self.weight, self.bias
    
    
    def train(self, iterations):
        for i in range(0, iterations):
            weight, bias = self.update_weights()
            
            cost = self.cost_function()
            self.cost_history.append(cost)
            self.iter_history.append(i)
            
            print('Iter: ' + str(i) + ', Weight: ' + str(weight) + ', Bias: ' + str(bias) + ', Error: ' + str(cost))
            
            if i % (iterations / 100) == 0:
                self.temp_plot(title = 'Learned Regression Plot', 
                    x_axis = 'x_training_data', 
                    y_axis = 'y_training_data')
            
            #time.sleep(1)
            
        return weight, bias

    
    def temp_plot(self, title, x_axis, y_axis):
        x_show = np.arange(0, 8)
        y_show = self.weight * x_show + self.bias
        
        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.scatter(x_train, y_train, marker = 'v', color = 'black')
        plt.plot(x_show, y_show, color = 'red')
        plt.show()