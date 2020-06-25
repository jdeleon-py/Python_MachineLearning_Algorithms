import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    '''
    # A simple model and multivariant will be combined into one model
    # if show_progress and (num_features == 1): show temp_plot

    #cost_function
    # -1 * np.sum((y * np.log10(p)) + ((1 - y) * np.log10(1 - p)))
    '''
    def __init__(self, X, y, rate, show_progress):
        '''
        # many of the model variables are defined and initialized
        # weights are vectorized and can support a simple and multivariate model
        # a cost_history and iteration_history log is defined to visualize an error plot
        '''
        self.X_train = X
        self.y_train = y
        self.learning_rate = rate
        self.show_progress = show_progress
        
        self.num_samples, self.num_features = self.X_train.shape
        
        self.weights = np.zeros(self.num_features)
        self.bias = 0
        
        self.cost_history = []
        self.iter_history = []
        
    
    def _sigmoid(self, x):
        '''
        # UTILITY FUNCTION
        
        # The sigmoid function transforms a linear continuous output into a discrete probablistic output
        # Domain: All real numbers, Range: (0, 1), At x = 0, f(x) = 0.5
        '''
        return 1 / (1 + np.exp(-x))

    
    def _predict(self, X_input):
        '''
        # UTILITY FUNCTION
        
        # As the model trains, y_hat is combined into calculating the sigmoid of the linear function of an input
        # f(x1, x2, ..., xn) = w1x1 + w2x2 + ... + wnxn
        # y_hat = sigmoid(f(x1, x2, ..., xn))
        '''
        linear_model = np.dot(X_input, self.weights) + self.bias
        y_predict = self._sigmoid(linear_model)
        return y_predict
    
    
    def cost_function(self, X, y):
        '''
        # cost function is combination of the cross entropy function (log loss function)
        
        # shorthand version:
        # Cost(y_hat, y_true) = -ln(y_hat),     if y_true = 1
        # Cost(y_hat, y_true) = -ln(1 - y_hat), if y_true = 0
        '''
        predictions = self._predict(self.X_train)
        cost = -1 * np.sum(((y * np.log(predictions)) + ((1 - y) * np.log(1 - predictions))))
        return cost
        
    
    def update_weights(self):
        '''
        # Weight vector and bias constant is updated via derivatives multiplied by learning_rate
        # Derivatives are equivalent to Linear Regression model derivatives 
        '''
        y_hat = self._predict(self.X_train)
        
        dw = (1 / self.num_samples) * np.dot(self.X_train.T, (y_hat - self.y_train))
        db = (1 / self.num_samples) * np.sum(y_hat - self.y_train)
            
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
        
        return self.weights, self.bias
    
    
    def train(self, iterations):
        '''
        # All functions are utilized here, where the user can observe the progress being made during training
        # The update_weights() function is put into an iterative loop, simultaneously while the error is being collected
        # Error can be observed to converge after a large number of iterations
        '''
        for i in range(0, iterations):
            weight, bias = self.update_weights()
            
            
            cost = self.cost_function(self.X_train, self.y_train)
            self.cost_history.append(cost)
            self.iter_history.append(i)
            
            if (self.show_progress == True) and (i % 1 == 0): # adjustable based on how many iterations
                print('Iter: ' + str(i) + ', Error: ' + str(cost))
                
            if self.show_progress and (self.num_features == 1):
                if i % (iterations / 100) == 0:
                    self.temp_plot(title = 'Learned Regression Plot', 
                        x_axis = 'x_training_data', 
                        y_axis = 'y_training_data')
            
        return weight, bias, cost
    
    
    def temp_plot(self, title, x_axis, y_axis):
        '''
        # Theoretically, this plot should display the data and draw a decision boundary
        # between the two sets of data
        '''
        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.scatter(x_train, y_train, marker = 'v', color = 'black')
        plt.plot(x_show, y_show, color = 'red')
        plt.show()
    
    
    def predict(self, X):
        '''
        # Unlike the _predict() utility function, the predict() function collapes into either a 1 or 0
        # rather than outputting a probability
        # This function is used for testing after training.
        '''
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        y_classify = [1 if i >= 0.5 else 0 for i in y_pred]
        return np.array(y_classify)
    
    
    def error_plot(self):
        '''
        # When plotted, the error will converge to a value as low as possible given the input data
        '''
        plt.title('Cost Function Visual')
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.plot(self.iter_history, self.cost_history, color = 'black')
        plt.show()
    
    
    def accuracy(self, y_true, y_pred):
        '''
        # This is an exploratory function that displays the accuracy of the model
        # Works when y_test is available to the user
        '''
        return np.sum(y_true == y_pred) / len(y_true)


