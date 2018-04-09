import time
import random
import numpy as np
from utils import *
from transfer_functions import * 
import matplotlib.pyplot as plt


class NeuralNetwork(object):
    
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, transfer_f=sigmoid, transfer_df=dsigmoid):
        """
        input_layer_size: number of input neurons
        hidden_layer_size: number of hidden neurons
        output_layer_size: number of output neurons
        iterations: number of iterations
        learning_rate: initial learning rate
        """

        # initialize transfer functions
        self.transfer_f = transfer_f
        self.transfer_df = transfer_df

        # initialize layer sizes
        self.input_layer_size = input_layer_size+1  # +1 for the bias node in the input Layer
        self.hidden_layer_size = hidden_layer_size+1 # +1 for the bias node in the hidden layer 
        self.output_layer_size = output_layer_size

        # initialize arrays for activations
        self.u_hidden = np.zeros(self.hidden_layer_size-1)
        self.u_output = np.zeros(self.output_layer_size)

        # initialize arrays for outputs
        self.o_input = np.ones(self.input_layer_size)
        self.o_hidden = np.ones(self.hidden_layer_size)
        self.o_output = np.ones(self.output_layer_size)

        # initialize arrays for partial derivatives according to activations
        self.dE_du_hidden = np.zeros(self.hidden_layer_size-1)
        self.dE_du_output = np.zeros(self.output_layer_size)

        # create randomized weights Yann LeCun method in 1988's paper (Default values)
        """input_range = 1.0 / self.input_layer_size ** (1/2)
        self.W_input_to_hidden = np.random.normal(loc = 0, scale = input_range, size =(self.input_layer_size, self.hidden_layer_size-1))
        self.W_hidden_to_output = np.random.uniform(size = (self.hidden_layer_size, self.output_layer_size)) / np.sqrt(self.hidden_layer_size)"""
        self.initialize()

    def initialize(self, wi=None, wo=None):
        input_range = 1.0 / self.input_layer_size ** (1/2)
        
        # initialize weights between input and hidden layer
        if wi is not None:
            self.W_input_to_hidden = wi
        else:
            self.W_input_to_hidden = np.random.normal(loc = 0, 
                                                      scale = input_range, 
                                                      size = (self.input_layer_size, 
                                                              self.hidden_layer_size-1))
            
        # initialize weights between hidden and output layer
        if wo is not None:
            self.W_hidden_to_output = wo
        else:
            self.W_hidden_to_output = np.random.uniform(size = (self.hidden_layer_size, 
                                                                self.output_layer_size)
                                                       ) / np.sqrt(self.hidden_layer_size)

    def train(self, train_data, validation_data, iterations=50, learning_rate=5.0, plot=True, batch_size=1):
        
        def next_batch(inputs, targets):#, batch_size):
            """
            Returns an iterable over dataset batches of size batch_size
            """
            
            for i in np.arange(0, len(inputs), batch_size):
                yield (inputs[i:(i+batch_size)%len(inputs)], targets[i:(i+batch_size)%len(inputs)])
        
        # if no batch size is specified, do stochastic gradient descent
        
        # initialize weights
        self.initialize()
        
        # get starting time to return execution time
        start_time = time.time()
        
        # get inputs and targets from the dataset
        inputs  = train_data[0]
        targets = train_data[1]
        
        best_acc_it = 0
        train_accuracies = []
        val_accuracies = []
        errors = []
        
        for it in range(iterations):
            errorsi = []
            count = 0
            for (inputs_batch, targets_batch) in next_batch(inputs, targets):
                
                # set derivatives to zero
                dE_dw_hidden = np.zeros(self.W_input_to_hidden.shape)
                dE_dw_output = np.zeros(self.W_hidden_to_output.shape)
                
                # compute the derivatives over the batch
                for i in range(len(inputs_batch)):

                    # compute the outputs
                    self.feedforward(inputs_batch[i])

                    # compute the derivatives
                    dE_dw_hidden_b, dE_dw_output_b = self.backpropagate(targets_batch[i])
                    
                    # update the total derivatives
                    dE_dw_hidden += dE_dw_hidden_b
                    dE_dw_output += dE_dw_output_b

                    # compute the squared error
                    error = targets_batch[i] - self.o_output
                    error *= error
                    errorsi.append(error)

                    # compute training data accuracy
                    target = np.argmax(targets_batch[i])
                    prediction = np.argmax(self.o_output)
                    if target == prediction:
                        count += 1
                
                # average the derivatives over the batch
                dE_dw_hidden /= batch_size
                dE_dw_output /= batch_size
                
                # update the weights
                self.update_weights(dE_dw_hidden, dE_dw_output, learning_rate)
               
            # keep track of lerning values
            errors.append(np.average(errorsi))            
            train_accuracies.append(count * 100 / len(inputs))
            new_accuracy = self.accuracy(validation_data)
            val_accuracies.append(new_accuracy)
            
            # update best accuracy
            if new_accuracy > val_accuracies[best_acc_it]:
                best_acc_it = it
        
        # plot learning curves and best accuracy
        if plot:
            self.plot_curves(train_accuracies, 
                             val_accuracies, 
                             errors, 
                             learning_rate, 
                             best_acc_it)
        
        return val_accuracies[best_acc_it], time.time() - start_time
       
    def train_xe(self, data, validation_data, iterations=50, learning_rate=5.0, verbose=False):
        start_time = time.time()
        training_accuracies = []
        validation_accuracies = []
        errors = []
        xes = []
        inputs  = data[0]
        targets = data[1]
        best_val_acc = 100*self.predict(validation_data)/len(validation_data[0])
        best_i2h_W = self.W_input_to_hidden
        best_h2o_W = self.W_hidden_to_output
        for it in range(iterations):
            self.feedforward_xe(inputs)
            self.backpropagate_xe(targets, learning_rate=learning_rate)
            xe = targets*np.log(self.o_output)*(-1)
            error = targets - self.o_output
            error *= error
            training_accuracies.append(100*self.predict(data))
            validation_accuracies.append(100*self.predict(validation_data))
            if validation_accuracies[-1] > best_val_acc:
                best_i2h_W = self.W_input_to_hidden
                best_h2o_W = self.W_hidden_to_output
            if verbose:
                print("[Iteration %2d/%2d]  -Training_Accuracy:  %2.2f %%  -Validation_Accuracy: %2.2f %%  -time: %2.2f " %(it+1, iterations,
                                                            training_accuracies[-1], validation_accuracies[-1], time.time() - start_time))
                print("    - MSE:", np.sum(error)/len(targets))
                print("    - X-Entropy:", np.sum(xe)/len(targets))
        print("Training time:", time.time()-start_time)
        self.W_input_to_hidden = best_i2h_W
        self.W_hidden_to_output = best_h2o_W
        plot_train_val(range(1, iterations+1), training_accuracies, validation_accuracies, "Accuracy")

    def accuracy(self, test_data):
        """ Returns percentage of well classified samples """
        
        count = 0
        
        # iterate over the dataset
        for i in range(len(test_data[0])):
            
            # compute the predictions
            self.feedforward(test_data[0][i])
            
            # count correct predictions
            target = np.argmax(test_data[1][i])
            prediction = np.argmax(self.o_output)
            if target == prediction:
                count += 1
                
        return count * 100 / len(test_data[0])
    
    def plot_curves(self, train_accuracies, val_accuracies, errors, learning_rate, best_acc_it):
        
        # get x axis
        iterations = np.arange(len(train_accuracies))
        
        _, ax = plt.subplots(1, 2, figsize=(15, 7))
        
        # plot accuracies curve
        ax[0].plot(iterations, train_accuracies, label="Training data", color="blue")
        ax[0].plot(iterations, val_accuracies, label="Validation data", color="orange")
        best_accuracy = val_accuracies[best_acc_it]
        ax[0].plot([0, best_acc_it, best_acc_it], [best_accuracy, best_accuracy, 0], color="green", ls="--", alpha=0.6, label="Best accuracy")
        ax[0].legend()
        ax[0].grid()
        ax[0].set_title("Accuracy learning curve")
        ax[0].set_xlabel("Iterations")
        ax[0].set_ylabel("Accuracy [%]")
        
        # plot MSE curve
        ax[1].plot(iterations, errors, label="Mean Squared Error")
        ax[1].legend()
        ax[1].grid()
        ax[1].axhline(y=0, color="red", ls="--", alpha=0.5)
        ax[1].set_title("Mean Squared Error learning curve")
        ax[1].set_xlabel("Iterations")
        ax[1].set_ylabel("MSE")
        
        plt.suptitle('Hidden Layer Size: {} - Learning Rate: {} - Accuracy: {} %'.format(self.hidden_layer_size-1, learning_rate, best_accuracy))
        plt.show() 

