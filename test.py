# Test script

import nn_rnd_training
import numpy as np
from scipy.special import expit # sigmoid

config = {
    'num_input':            3,     # number of input datapoints
    'sequence_input':       True,  # whether the input data contains data points for each timestep
    'net_input_add_onehot': False, # whether to add a one_hot encoding layer to the input. Only if sequence_input is true
    'num_hidden':           30,    # number of hidden neurons, excluding output neurons
    'num_output':           2,     # number of output neurons
    'sequence_output':      False, # whether the output should contain the output at each timestep
    'timesteps':            4,     # number of iterations to do before reading the output
    'net_add_softmax':      False, # whether to add a softmax layer at the end
    'loss':                        # the loss function
        'mean_squared_error',
    'test_stddev':          0.01,  # stddev for weight adjustments for the random step
    'batch_size':           1000,  # this many points are tested around the current point in the weight space
    'epoch_steps':          400,   # steps in an epoch
    'epochs':               10000, # number of epochs
    'debug':                False  # whether to print tensors during runs
}

# Data generator
# Generate random training data
class DataIterator:
    """Data generator"""
    
    def __init__(self):
        pass
    
    def __next__(self):
        
        data = np.random.random((config['batch_size'], config['timesteps'], config['num_input'])) * 2. - 1.
        data[:,1,:] = data[:,0,:] + 1
        data[:,2,:] = data[:,0,:] + 2
        data[:,3,:] = data[:,0,:] + 3
        
        target = np.ones((config['batch_size'], config['num_output']))
        
        target[:,0] = data[:,0,0] + data[:,0,1]
        target[:,1] = data[:,0,0] + data[:,0,2]
        target = expit(target)
        
        return (data, target)        

nn_rnd_training.NNRndTraining(config).train(DataIterator())
