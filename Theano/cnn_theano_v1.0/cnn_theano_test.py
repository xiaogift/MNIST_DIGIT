import pickle_data as pickle
import cnn_theano as cnn

training_data, validation_data, test_data = pickle.load_data()
training_data   = cnn.tranfer_2shared_data(training_data)
validation_data = cnn.tranfer_2shared_data(validation_data)
test_data	    = cnn.tranfer_2shared_data(test_data)

net = cnn.Network([
	# --- image_shape | filter_shape | poolsize | activation_fn
    # --- First layer - (10, 1, 28, 28) pic_num/mini_size, RGB_tunnel, pic_height, pic_width
    # --- Feature Map / Weight - (20, 1,5,5) conv_core_num, 3_layer_features_num, height, width 

	# cnn.ConvPoolLayer((10, 1, 28, 28),(20, 1,5,5),(2, 2), cnn.ReLU),
	# cnn.ConvPoolLayer((10,20, 12, 12),(40,20,5,5),(2, 2), cnn.ReLU),
	# cnn.FullyConnectedLayer(40*4*4, 100, cnn.ReLU),

    cnn.ConvPoolLayer((10,1, 28, 28),(20,1,5,5),(2, 2), cnn.ReLU),
    cnn.FullyConnectedLayer(20*12*12, 100),
	cnn.SoftmaxLayer(100, 10)], 10)
net.SGD(training_data, 30, 10, 0.03, validation_data, test_data, lmbda=0.1)

"""
Training mini-batch number 0.0
Training mini-batch number 1000.0 
Training mini-batch number 2000.0
Training mini-batch number 3000.0
Training mini-batch number 4000.0
Epoch 0: validation accuracy 88.41%
This is the best validation accuracy to date.
The corresponding test accuracy is 87.47%
Training mini-batch number 5000.0
Training mini-batch number 6000.0
Training mini-batch number 7000.0
Training mini-batch number 8000.0
Training mini-batch number 9000.0
Epoch 1: validation accuracy 89.95%
This is the best validation accuracy to date.
The corresponding test accuracy is 89.76%
Training mini-batch number 10000.0
Training mini-batch number 11000.0
Training mini-batch number 12000.0
Training mini-batch number 13000.0
Training mini-batch number 14000.0
Epoch 2: validation accuracy 91.29%
This is the best validation accuracy to date.
The corresponding test accuracy is 90.53%
Training mini-batch number 15000.0
Training mini-batch number 16000.0
Training mini-batch number 17000.0
Training mini-batch number 18000.0
Training mini-batch number 19000.0
Epoch 3: validation accuracy 92.10%
This is the best validation accuracy to date.
The corresponding test accuracy is 91.61%
Training mini-batch number 20000.0
Training mini-batch number 21000.0
Training mini-batch number 22000.0
Training mini-batch number 23000.0
Training mini-batch number 24000.0
Epoch 4: validation accuracy 92.08%
Training mini-batch number 25000.0
Training mini-batch number 26000.0
Training mini-batch number 27000.0
Training mini-batch number 28000.0
Training mini-batch number 29000.0
Epoch 5: validation accuracy 92.87%
This is the best validation accuracy to date.
The corresponding test accuracy is 92.49%
Training mini-batch number 30000.0
Training mini-batch number 31000.0
Training mini-batch number 32000.0
Training mini-batch number 33000.0
Training mini-batch number 34000.0
Epoch 6: validation accuracy 93.25%
This is the best validation accuracy to date.
The corresponding test accuracy is 92.97%
Training mini-batch number 35000.0
Training mini-batch number 36000.0
Training mini-batch number 37000.0
Training mini-batch number 38000.0
Training mini-batch number 39000.0
Epoch 7: validation accuracy 93.90%
This is the best validation accuracy to date.
The corresponding test accuracy is 93.43%
Training mini-batch number 40000.0
Training mini-batch number 41000.0
"""

"""
Traceback (most recent call last):
  File "D:\installation\Python\lib\site-packages\theano\compile\function_module.py", line 884, 
    in __call__self.fn() if output_subset is None else\
MemoryError

During handling of the above exception, another exception occurred:
Traceback (most recent call last):
  File "D:\WORKSPACE\Python\Digital-Recognition\network\network_convolution_test.py", line 22, 
    in <module> net.SGD(training_data, 30, 10, 0.03, validation_data, test_data, lmbda=0.1)
  File "D:\WORKSPACE\Python\Digital-Recognition\network\network_convolution.py", line 72, 
    in SGD train_min_batch(minibatch_index)
  File "D:\installation\Python\lib\site-packages\theano\compile\function_module.py", line 898, 
    in __call__storage_map=getattr(self.fn, 'storage_map', None))
  File "D:\installation\Python\lib\site-packages\theano\gof\link.py", line 325, 
    in raise_with_opreraise(exc_type, exc_value, exc_trace)
  File "D:\installation\Python\lib\site-packages\six.py", line 685, 
    in reraise raise value.with_traceback(tb)
  File "D:\installation\Python\lib\site-packages\theano\compile\function_module.py", line 884, 
    in __call__self.fn() if output_subset is None else\

MemoryError: 
Apply node that caused the error: Elemwise{sqr,no_inplace}(w)
Toposort index: 8
Inputs types: [TensorType(float32, matrix)]
Inputs shapes: [(2880, 100)]
Inputs strides: [(400, 4)]
Inputs values: ['not shown']
Outputs clients: [[Sum{acc_dtype=float64}(Elemwise{sqr,no_inplace}.0)]]

HINT: Re-running with most Theano optimization disabled could give you a back-trace of when this node was created. 
This can be done with by setting the Theano flag 'optimizer=fast_compile'. 
If that does not work, Theano optimizations can be disabled with 'optimizer=None'.

HINT: Use the Theano flag 'exception_verbosity=high' for a debugprint and storage map footprint of this apply node.
[Finished in 1695.4s with exit code 1]
[shell_cmd: python -u "D:\WORKSPACE\Python\Digital-Recognition\network\network_convolution_test.py"]
[dir: D:\WORKSPACE\Python\Digital-Recognition\network]
"""
