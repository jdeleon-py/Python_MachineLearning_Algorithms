import numpy as np

class Standardize:
    '''
    adapted from Kenzo's Blog --- http://kenzotakahashi.github.io/scikit-learns-useful-tools-from-scratch.html
    '''
    def __init__(self, X):
        '''
        # related to the statistical idea of a z-score
        # standardized data to make the data more relatable to each other...
        # As a result, the model algortihm is a lot faster.
        '''
        self.input_data = X
        
        self.mean = np.mean(self.input_data, axis=0)
        self.scale = np.std(self.input_data - self.mean, axis=0)

    
    def fit_transform(self):
        '''
        # This function combines _fit() and _transform() to standardize the data, which speeds up the algorithm
        # Note: do not standardize binary values, only standardize continuous values
        # Note: both train and test data need to be standardized together when processed together
        '''
        return (self.input_data - self.mean) / self.scale