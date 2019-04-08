# nn_rnd_training

This module trains a completely free neural network (a single repeated fully connected layer)
using a smart random walk of the weight space to avoid the vanishing/exploding gradients problem.
This is largely equivalent to using a genetic algorithm (evolution / mutations) to optimise the weights (except gene exchange).

It uses TensorFlow.

A **completely free network** has a number of neurons with arbitrary
connections. This is modelled using a weight matrix that maps an
`input+bias+activations` vector to an `activations` vector,
and is applied a set number of times before the output is read.
This can model recursion as well as multiple levels of a feedforward network.
The output is a (prefix) slice of the activations; that is, the activations vector
is separated into `output+hidden` states.

         From:
         bias+input   output+hidden
     To: +-----------+--------------+
    out  |           |              |
    hid  |           |              |
         +-----------+--------------+

While a completely free neural model
(i.e. single repeated interconnected layer) could be trained with backprop, 
that would still suffer from vanishing/exploding gradients
due to the weight matrix being applied repeatedly.
This is true even if nonlinearities with derivatives close to 1 are used, like ReLUs:
see e.g. Rohan Saxena's answer at https://www.quora.com/What-is-the-vanishing-gradient-problem .

To avoid this, we use a "smart" random walk to approximate the gradient in the weight space.
We run the graph for each datapoint in the batch,
and then run it with slightly adjusted weights on the same datapoint.
This adjustment is different for each datapoint in the batch.
Then we calculate the change in the loss,
and multiply the negative change with the adjustment
(if the change is good, we want to go in that direction,
if it's bad, in the opposite direction),
then update the weights using the averages of these multiplied adjustments across the batch.

# Usage

```python
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
```

# Todo

- Implement an evaluation mode; saving, loading weights
- Distribute to multiple GPUs
- Convert to a trainer on an arbitrary graph (on a collection of trainable Variables)
- Allow special, limited architectures
- Change the learning rate using a variable, and variable.load https://www.tensorflow.org/versions/r1.5/api_docs/python/tf/Variable#load

