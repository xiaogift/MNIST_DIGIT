'''
Alorithm: Convolutional Neural Network (CNN)
Network input : 2D data
Network output: 1D data
'''

#============================== Convolutional Neural Network ==============================#
import theano, pickle, numpy as np
import theano.tensor as T
import theano.tensor.nnet as nn
import theano.tensor.signal.pool as pool
import theano.tensor.shared_randomstreams as shared_randomstreams

# try: theano.config.device = 'gpu'
# except: pass

# Damn you CPU config !
# theano.config.floatX = 'float32'
fx = theano.config.floatX

class Network:

    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.params = [param for layer in self.layers for param in layer.params]
        self.x      = T.matrix('x')
        self.y      = T.ivector('y')
        init_layer  = self.layers[0]
        init_layer.set_inpt(self.x, self.x, mini_batch_size)
        for j in range(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(prev_layer.output, prev_layer.output_dropout, mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta, validation_data, test_data, lmbda=0.0):
        training_x, training_y     = training_data
        validation_x, validation_y = validation_data
        test_x, test_y             = test_data
        num_training_batches   = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches       = size(test_data)/mini_batch_size

        l2_norm = sum([(layer.w**2).sum() for layer in self.layers])
        cost    = self.layers[-1].cost(self)+ 0.5*lmbda*l2_norm/num_training_batches
        grads   = T.grad(cost, self.params)
        updates = [(param, param-eta*grad) for param, grad in list(zip(self.params, grads))]

        i = T.lscalar() # mini-batch index
        train_min_batch      = theano.function( inputs = [i], outputs= cost,  updates=updates, 
            givens={ self.x: training_x[i*mini_batch_size: (i+1)*mini_batch_size],
                     self.y: training_y[i*mini_batch_size: (i+1)*mini_batch_size] })
        validate_mb_accuracy = theano.function( inputs = [i], outputs= self.layers[-1].accuracy(self.y), 
            givens={ self.x: validation_x[i*mini_batch_size: (i+1)*mini_batch_size],
                     self.y: validation_y[i*mini_batch_size: (i+1)*mini_batch_size] })
        test_mb_accuracy     = theano.function( inputs = [i], outputs= self.layers[-1].accuracy(self.y), 
            givens={ self.x: test_x[i*mini_batch_size: (i+1)*mini_batch_size],
                     self.y: test_y[i*mini_batch_size: (i+1)*mini_batch_size] })
        
        # self.test_mb_predictions = theano.function( inputs = [i], outputs= self.layers[-1].y_out, 
        #     givens={ self.x: test_x[i*mini_batch_size: (i+1)*mini_batch_size] })

        best_validation_accuracy = 0.0
        for epoch in range(epochs):
            for minibatch_index in range(int(num_training_batches)):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0: print('Training mini-batch number {0}'.format(iteration))
                # cost_ij = train_min_batch(minibatch_index)
                train_min_batch(minibatch_index)
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean([validate_mb_accuracy(j) for j in range(int(num_validation_batches))])
                    print('Epoch {0}: validation accuracy {1:.2%}'.format(epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print('This is the best validation accuracy to date.')
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean( [test_mb_accuracy(j) for j in range(int(num_test_batches))])
                            print('The corresponding test accuracy is {0:.2%}'.format(test_accuracy))
        print('Finished training network.')
        print('Best validation accuracy of {0:.2%} obtained at iteration {1}'.format(best_validation_accuracy, best_iteration))
        print('Corresponding test accuracy of {0:.2%}'.format(test_accuracy))

#============================== Layers ==============================#
class ConvPoolLayer:

    def __init__(self, image_shape, filter_shape, poolsize=(2, 2), activation_fn=nn.sigmoid):
        self.image_shape   = image_shape
        self.filter_shape  = filter_shape
        self.poolsize      = poolsize
        self.activation_fn = activation_fn
        n_out  = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = share(rand(size=filter_shape, scale=np.sqrt(1.0/n_out)), 'w')
        self.b = share(rand(size=(filter_shape[0],), scale=1.0), 'b')
        self.params = [self.w, self.b]
        
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt   = inpt.reshape(self.image_shape)
        conv_out    = nn.conv.conv2d(input=self.inpt, filters=self.w, filter_shape=self.filter_shape, image_shape=self.image_shape)
        pooled_out  = pool.pool_2d(conv_out,self.poolsize, ignore_border=True)
        self.output = self.activation_fn(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output

class FullyConnectedLayer:

    def __init__(self, n_in, n_out, activation_fn=nn.sigmoid, p_dropout=0.0):
        self.n_in  = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        self.w = share( rand(size=(n_in, n_out), scale=np.sqrt(1.0/n_out)), 'w' )
        self.b = share( rand(size=(n_out,), scale=1.0), 'b' )
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt   = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out  = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y): return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer:

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in  = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        self.w = share( np.zeros((n_in, n_out), dtype=fx), 'w')
        self.b = share( np.zeros((n_out,), dtype=fx), 'b')
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt   = inpt.reshape((mini_batch_size, self.n_in))
        self.output = nn.softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out  = T.argmax(self.output, axis=1)
        self.inpt_dropout   = dropout_layer( inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = nn.softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net): return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])
    def accuracy(self, y): return T.mean(T.eq(y, self.y_out))

#============================== Miscellanea ==============================#
def ReLU(z): return T.maximum(0.0, z)
def linear(z): return z
def size(data): return data[0].get_value(borrow=False).shape[0]

def rand(size, scale=1.0): return np.asarray(np.random.normal(loc=0.0, scale=1.0, size=size), dtype=fx)
def share(data, name): return theano.shared( value=data, name=name, borrow=False)

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams( np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, fx)

def tranfer_2shared_data(data):
    shared_x = share(np.asarray(data[0], dtype=fx), 'x')
    shared_y = share(np.asarray(data[1], dtype=fx), 'y')
    return shared_x, T.cast(shared_y, 'int32')

#============================== XIAOLI-20170416 ==============================#