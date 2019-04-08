import tensorflow as tf
import time

class NNRndTraining:
    
    def __init__(self, config):
        """
            config = {
                'num_input':            3,     # number of input datapoints
                'sequence_input':       False, # whether the input data contains data points for each timestep
                'net_input_add_onehot': False, # whether to add a one_hot encoding layer to the input. Only if sequence_input is true
                'num_hidden':           30,    # number of hidden neurons, excluding output neurons
                'num_output':           2,     # number of output neurons
                'sequence_output':      False, # whether the output should contain the output at each timestep
                'timesteps':            100,   # number of iterations to do before reading the output
                'net_add_softmax':      False, # whether to add a softmax layer at the end
                'loss':                        # the loss function
                    'mean_squared_error'
                    'categorical_crossentropy'
                'test_stddev':          0.01,  # stddev for weight adjustments for the random step
                'batch_size':           1000,  # this many points are tested around the current point in the weight space
                'epoch_steps':          400,   # steps in an epoch
                'epochs':               10000, # number of epochs
                'debug':                False  # whether to print tensors during runs
            }
        """

        self.config = config

        # From: input + bias + activations  == input + bias + hidden + output
        self.config['num_from'] = self.config['num_input'] + 1 + self.config['num_hidden'] + self.config['num_output']

        #   To: activations (incl. output)  == hidden + output
        self.config['num_to'] = self.config['num_hidden'] + self.config['num_output']
        
        # weights <from, to>
        # initializer=tf.random_normal_initializer()
        self.W = tf.get_variable("W", shape=(self.config['num_from'], self.config['num_to']), trainable=True)
        
        # output variable
        # if self.config['sequence_output']:
        self.initial_output = tf.zeros((self.config['batch_size'], 0, self.config['num_output']))
        
        # random adjustments <batch_size, from, to>
        # We don't broadcast here to apply different random values to each datapoint in the batch
        self.W_adj_source = tf.random_normal(shape=(self.config['batch_size'], self.config['num_from'], self.config['num_to']), stddev=self.config['test_stddev'])
        
        # activations <batch_size, to>
        # Each datapoint in the batch is run separately so has separate activations
        self.initial_activations = tf.zeros((self.config['batch_size'], self.config['num_to'])) # tf.get_variable("activations", shape=(self.config['num_to'],), initializer=tf.zeros_initializer())
        
        # bias <batch_size, 1>
        self.bias = tf.ones((self.config['batch_size'], 1))
        
        self.initial_i = tf.constant(0)


    def analyze_config(self):
        num_weights = self.config['num_from'] * self.config['num_to']
        print("Number of weights: {}  Test coverage: {:.2f}%".format(num_weights, self.config['batch_size']/num_weights/2*100))


    def setup_print(self, t, message):
        """Optionally add a computation node to display a tensor"""
        if self.config['debug']:
            return tf.Print(t, (t,), message=message+": ", summarize=10)
        return t


    def setup_forward(self, W, input_data, prefix=""):
        """Create the graph for one forward step
            W - weights tensor <batch_size, from, to>
            input_data - tensor <batch_size, num_input> 
                or <batch_size, timesteps, num_input> if sequence_input
                or <batch_size, timesteps> if sequence_input & net_input_add_onehot
            prefix - string
        """
        
        def loop_body(i, activations, outputcollect):
            
            if self.config['sequence_input']:
                # Cut out the correct input
                if self.config['net_input_add_onehot']:
                    inp = tf.slice(input_data, (0,i), (self.config['batch_size'], 1), name=prefix+"/inputSlice") # <batch_size, 1>
                    inp = tf.squeeze(inp, 1, name=prefix+"/inputSqueeze") # <batch_size>
                    inp = tf.one_hot(indices=inp, depth=self.config['num_input']) # <batch_size, num_input>
                else:
                    inp = tf.slice(input_data, (0,i,0), (self.config['batch_size'], 1, self.config['num_input']), name=prefix+"/inputSlice") # <batch_size, 1, num_input>
                    inp = tf.squeeze(inp, 1, name=prefix+"/inputSqueeze") # <batch_size, num_input>
            else:
                inp = input_data
            inp = self.setup_print(inp, "input data")
            
            # Concatenate input, bias, activations
            inp = tf.concat([inp, self.bias, activations], axis=1, name=prefix+"/stepconcat") # <batch_size, from>
            inp = tf.expand_dims(inp, 1) # <batch_size, 1, from>
            
            # Fully connected
            # <batch_size, 1, to> <= <batch_size, 1, from> @ <batch_size, from, to>
            activations = tf.matmul(inp, W, name=prefix+"/stepmatmul")
            activations = tf.squeeze(activations, 1) # <batch_size, to>
        
            # Leaky ReLU
            # This allows values to blow up
            ## activations = tf.maximum(activations, activations * .3, name=prefix+"/lrelu")
            
            # Sigmoid
            activations = tf.sigmoid(activations) # <batch_size, to>
            
            # Store the output if we need outputs from all timesteps
            # Alternative may be: https://stackoverflow.com/questions/39157723/how-to-do-slice-assignment-in-tensorflow/43139565#43139565
            if self.config['sequence_output']:
                output = tf.slice( # -> <batch_size, output>
                    activations, 
                    (0,0), 
                    (self.config['batch_size'], self.config['num_output']), 
                    name=prefix+"/outputslice"
                )
                output = tf.expand_dims(output, axis=1) # <batch_size, 1, output>
                outputcollect = tf.concat([outputcollect, output], axis=1)
            
            return tf.add(i,1), activations, outputcollect
        
        loop_out = tf.while_loop(
            cond=(lambda
                    i, 
                    activations,
                    outputcollect:
                tf.less(i, self.config['timesteps'])
            ),
            body=loop_body,
            loop_vars=[
                    self.initial_i,
                    self.initial_activations,
                    self.initial_output
            ],
            shape_invariants=[
                    self.initial_i.get_shape(),
                    self.initial_activations.get_shape(),
                    tf.TensorShape([self.config['batch_size'], None, self.config['num_output']])
            ],
            back_prop=False,
            # return_same_structure=True,
            name=prefix+"/loop"
        )
        
        # Get the output
        if self.config['sequence_output']:
            output = loop_out[2]
            # Set shape otherwise broadcasting messes this up
            output.set_shape((self.config['batch_size'], self.config['timesteps'], self.config['num_output']))
        else:
            activations = loop_out[1] # <batch_size, to>
            output = tf.slice( # -> <batch_size, output>
                activations, 
                (0,0), 
                (self.config['batch_size'], self.config['num_output']), 
                name=prefix+"/outputslice"
            )

        if self.config['net_add_softmax']:
            # tf.nn.softmax
            output = tf.exp(output) / tf.expand_dims(tf.reduce_sum(tf.exp(output), axis=-1), axis=-1)
        
        return output
    

    def setup_loss(self, output, target, prefix=""):
        """Create graph for calculating the loss.
            output - tensor <batch_size, (timesteps,) output>
            target - tensor <batch_size, output>
                or <batch_size, timesteps, output> if sequence_output
                or <batch_size, timesteps> if sequence_output & net_target_add_onehot
            prefix - string
            
            Returns <batch_size>
        """
        if self.config['sequence_output']:
            if self.config['net_target_add_onehot']:
                target = tf.one_hot(indices=target, depth=self.config['num_output'])
            axes = [1,2]
        else:
            axes = 1

        if self.config['loss'] == 'mean_squared_error':
            return tf.reduce_sum(tf.square(output - target), axis=axes) # sum of the squares <batch_size>
        if self.config['loss'] == 'categorical_crossentropy':
            # https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks
            return (- tf.reduce_sum(target * tf.log(output), axis=axes))
 
        raise Exception('Unknown loss')


    def setup_train(self, input_data, target):
        """Create graph for training on one batch
        """
        
        W_my = self.setup_print(self.W, "intial W")
        
        # The weights with the random adjustment are <batch_size, from, to>, so
        # we inflate W here, too.
        W_exp = tf.tile(tf.expand_dims(W_my, 0), [self.config['batch_size'], 1, 1]) # <batch_size, from, to>

        # 1. Actual output
        output = self.setup_forward(W_exp, input_data, prefix="org") # <batch_size, (timesteps,) output>
        loss = self.setup_loss(output, target, prefix="org") # <batch_size>
        loss = self.setup_print(loss, "loss")
        
        # 2. Test output in the environment
        # TODO Do the random test around the decayed weights
        # NOTE: W_adj_source keeps its value inside a single run
        # https://stackoverflow.com/questions/52213325/are-tensorflow-random-values-guaranteed-to-be-the-same-inside-a-single-run
        W_adj = self.W_adj_source # <batch_size, from, to>
        W_adj = self.setup_print(W_adj, "W_adj")
        
        output_adj = self.setup_forward(W_exp + W_adj, input_data, prefix="adj")
        loss_adj = self.setup_loss(output_adj, target, prefix="adj")
        loss_adj = self.setup_print(loss_adj, "loss_adj")
        # improvement is positive when we go from large error to small error
        improvement = loss - loss_adj # <batch_size>
        improvement = self.setup_print(improvement, "improvement")
        
        # Update the weights
        improvement = tf.expand_dims(tf.expand_dims(improvement, 1), 2) # <batch_size, 1, 1>
        weight_update = W_adj * improvement # <batch_size, from, to>
        weight_update = self.setup_print(weight_update, "weight_update")
        weight_update = tf.reduce_mean(weight_update, axis=0) # <from, to>
        
        weight_update = self.setup_print(weight_update, "weight_update_reduced")
        weight_update = self.W.assign_add(weight_update)
        
        # Get the average loss
        loss_avg = tf.reduce_mean(loss, axis=0)
        
        return weight_update, loss_avg

    
    def train(self, data_iterator):
        """
            data_iterator -- should return the input data and the target in each step
                input data:  <batch_size, num_input>
                             or <batch_size, timesteps, num_input> if sequence_input
                             or <batch_size, timesteps> if sequence_input && net_input_add_onehot
                output data: <batch_size, num_output>
                             or <batch_size, timesteps, num_output> if sequence_output
                             or <batch_size, timesteps> if sequence_output && net_target_add_onehot
        """
        
        if self.config['sequence_input']:
            if self.config['net_input_add_onehot']:
                input_data_ph = tf.placeholder(tf.uint8, shape=(self.config['batch_size'], self.config['timesteps']))
            else:
                input_data_ph = tf.placeholder(tf.float32, shape=(self.config['batch_size'], self.config['timesteps'], self.config['num_input']))
        else:
            input_data_ph = tf.placeholder(tf.float32, shape=(self.config['batch_size'], self.config['num_input']))
        
        if self.config['sequence_output']:
            if self.config['net_target_add_onehot']:
                target_ph = tf.placeholder(tf.uint8, shape=(self.config['batch_size'], self.config['timesteps']))
            else:
                target_ph = tf.placeholder(tf.float32, shape=(self.config['batch_size'], self.config['timesteps'], self.config['num_output']))
        else:
            target_ph = tf.placeholder(tf.float32, shape=(self.config['batch_size'], self.config['num_output']))
        
        training, loss_avg_t = self.setup_train(input_data_ph, target_ph)
        
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        
        self.analyze_config()
        
        for epoch in range(self.config['epochs']):
            starttime = time.time()
            for step in range(self.config['epoch_steps']):
                input_data, target = next(data_iterator)
                tmp, loss_avg_value = session.run([training, loss_avg_t], {input_data_ph:input_data, target_ph:target})
            print("Epoch: {} Loss: {} Elapsed:{}s".format(epoch, loss_avg_value, (time.time() - starttime)))
