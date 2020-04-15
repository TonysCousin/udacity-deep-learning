import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                    (self.input_nodes, self.hidden_nodes))
        #print("init: input_nodes = ", input_nodes, ", hidden_nodes = ", hidden_nodes)
        #print("init: weights_input_to_hidden = ")
        #print(self.weights_input_to_hidden)

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        #self.activation_function = lambda x : 1.0 / (1.0 + np.exp(-x)) #jas
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))
        self.activation_function = sigmoid
                    

        #jas Compute the derivative in a more general way so the code below doesn't have to assume what the 
        #    activation function is.
        def sigmoid_derivative(x):
            s = self.activation_function(x)
            res = s * (1 - s)
            print("activation_derivative result =")
            print(res)
            return res
        self.activation_derivative = sigmoid_derivative
        
    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        
        #jas
        #print("train:  features.shape = ", features.shape)
        #print("input_nodes = ", self.input_nodes, ", hidden_nodes = ", self.hidden_nodes)
        #print("weights_input_to_hidden.shape = ", self.weights_input_to_hidden.shape)
        #print(self.weights_input_to_hidden)
        #print("weights_hidden_to_output.shape = ", self.weights_hidden_to_output.shape)
        #print(self.weights_hidden_to_output)
        
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)
        
        #jas
        #print("train results: n_records = ", n_records)
        #print("weights_input_to_hidden, weights_hidden_to_output =")
        #print(self.weights_input_to_hidden)
        #print(self.weights_hidden_to_output)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        
        #jas - all of this method
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs #using linear activation function in the output layer
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        #jas - all below
        error = y - final_outputs
        #print("error = ")
        #print(error)
        
        # hidden layer's contribution to the error
        output_error_term = error * 1.0 #output layer activation function is linear, so derivative is 1
        #print("output_error_term = ")
        #print(output_error_term)
        hidden_error = np.matmul(self.weights_hidden_to_output, output_error_term)
        #print("hidden_error = ")
        #print(hidden_error)
        
        #hidden_error_term = hidden_error * self.activation_derivative(hidden_outputs) #this gives slightly wrong answers
        hidden_error_term = hidden_error * hidden_outputs * (1.0 - hidden_outputs) #this passes unit test run_train
        #print("hidden_error_term =")
        #print(hidden_error_term)
        
        # Weight step (input to hidden)
        delta_weights_i_h += self.lr * hidden_error_term * X[:,None]
        #print("delta_weights_i_h =")
        #print(delta_weights_i_h)
        
        # Weight step (hidden to output)
        delta_weights_h_o += self.lr * output_error_term * hidden_outputs[:,None]
        #print("delta_weights_h_o = ")
        #print(delta_weights_h_o)
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        #jas
        self.weights_hidden_to_output += delta_weights_h_o / n_records
        self.weights_input_to_hidden += delta_weights_i_h / n_records

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        #jas - all below
        #print("Enter run: features, weigts_input_to_hidden =")
        #print(features)
        #print(self.weights_input_to_hidden)
        hidden_inputs = np.matmul(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        #print("hidden_inputs, hidden_outputs = ")
        #print(hidden_inputs)
        #print(hidden_outputs)
        
        final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs #using linear activation function in the output layer
        #print("run output =")
        #print(final_outputs)
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 4000
learning_rate = 0.05
hidden_nodes = 4
output_nodes = 1

