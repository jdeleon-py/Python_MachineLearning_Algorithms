import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

class Multivariable_LinearRegression:
    '''
    # Designed for more than one feature, but can also operate with one feature,
    # therefore this is a universal algorithm.

    # Vector math and the numpy library apply to the algorithm's simplicity.

    # Weights are written as vectors, thus as many weights as the model requires can be operated on.
    # The number of samples and the number of features are unpacked from the shape of the feature matrix.
    '''

    def __init__(self, X, y, rate, show_progress):
        '''
        # X_train and y_train is initialized
        # Since the number of features could be more than 1, weight is a vector
        # bias and learning rate are also constants

        # cost history and iteration history is recorded in an array for the error plot
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
        
    
    def _predict(self, X_input):
        '''
        # UTILITY FUNCTION

        # y_hat = w1x1 + w2x2 + w3x3 + ... + wnxn + b
        '''

        y_predict = np.dot(X_input, self.weights) + self.bias
        return y_predict
    
    
    def cost_function(self, y_true, y_predicted):
        '''
        # Numpy library automatically calculates the vectorized MSE
        # error = sum[(y_true - y_pred) ** 2]
        '''

        return np.mean((y_true - y_predicted) ** 2)
    
    
    def update_weights(self):
        '''
        # Regression models are reinforced iteratively with a gradient descent algorithm.
        # In this case, the rate (derivative) of the weight vector and bias is defined as the derivative of the cost function.
        # A smaller learning rate is favorable to make the gradient descent more precise rather than chaotic.
        # Since an error as low as possible is optimal, the derivatives of weight and bias are subtracted from the current weight vector and bias.
        # The weight vector and bias of the features are updated per iteration.

        # Transpose of feature matrix is multiplied by the difference between y_pred and y_true
        '''

        y_hat = self._predict(self.X_train)
        
        dw = (1 / self.num_samples) * np.dot(2 * self.X_train.T, (y_hat - self.y_train))
        db = (1 / self.num_samples) * np.sum(2 * (y_hat - self.y_train))
            
        self.weights -= (self.learning_rate * dw)
        self.bias -= (self.learning_rate * db)
        
        return self.weights, self.bias
    
    
    def train(self, iterations):
        '''
        # The train() function puts the update_weights() function in an iterative loop.
        # With each iteration, data is collected and printed to screen.
        # Cost is appended to the cost_history array --- weights and bias is printed out and eventually returned.
        # Weight vectors are printed out as an array of weight indices
        '''

        for i in range(0, iterations):
            y_hat = self._predict(self.X_train)
            weight, bias = self.update_weights()
            
            cost = self.cost_function(self.y_train, y_hat)
            self.cost_history.append(cost)
            self.iter_history.append(i)
            
            if self.show_progress == True:
                print('Iter: ' + str(i) + ', Weight: ' + str(weight) + ', Bias: ' + str(bias) + ', Error: ' + str(cost))
            
        return weight, bias, cost
    
    
    def predict(self, X_input):
        '''
        # function to call after training
        # tests the data with the trained model
        '''

        y_predict = np.dot(X_input, self.weights) + self.bias
        return y_predict


if __name__ == '__main__':
    
    X, y = datasets.make_regression(n_samples=5000, n_features=5, noise=30, random_state=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=101)

    test = Multivariable_LinearRegression(X = X_train, y = y_train, rate = 0.001, show_progress = True)
    test.train(iterations = 25000)

    predicted = test.predict(X_test)
    #print(predicted)

