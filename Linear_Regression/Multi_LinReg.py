import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

class Multivariable_LinearRegression:
    '''
    Designed for more than one feature, but can also operate with one feature,
    therefore this is a universal algorithm.

    Vector math and the numpy library apply to the algorithm's simplicity 
    '''

    def __init__(self, X, y, rate, show_progress):
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
        y_predict = np.dot(X_input, self.weights) + self.bias
        return y_predict
    
    
    def cost_function(self, y_true, y_predicted):
        return np.mean((y_true - y_predicted) ** 2)
    
    
    def update_weights(self):
        y_hat = self._predict(self.X_train)
        
        dw = (1 / self.num_samples) * np.dot(2 * self.X_train.T, (y_hat - self.y_train))
        db = (1 / self.num_samples) * np.sum(2 * (y_hat - self.y_train))
            
        self.weights -= (self.learning_rate * dw)
        self.bias -= (self.learning_rate * db)
        
        return self.weights, self.bias
    
    
    def train(self, iterations):
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
        y_predict = np.dot(X_input, self.weights) + self.bias
        return y_predict


if __name__ == '__main__':
    
    X, y = datasets.make_regression(n_samples=5000, n_features=5, noise=30, random_state=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=101)

    test = Multivariable_LinearRegression(X = X_train, y = y_train, rate = 0.001, show_progress = True)
    test.train(iterations = 25000)

    predicted = test.predict(X_test)
    #print(predicted)

