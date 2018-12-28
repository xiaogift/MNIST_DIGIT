import numpy as np

#============================== Network ==============================#
class Network:

	def __init__(net, sizes):
		net.num_layers     = len(sizes)
		net.biases         = [np.random.randn( x, 1 ) for x in sizes[1:]]
		net.weights        = [np.random.randn( x, y ) for y, x in list(zip(sizes[:-1], sizes[1:]))]
		net.epoch_amount   = 10
		net.min_batch_size = 10
		net.learning_rate  = 3.0
		net.results_list   = []

	def training(net, training_data):
		print('===== Training Started =====')
		data_size = len(training_data)
		for index in range(net.epoch_amount):
			np.random.shuffle(training_data)
			min_batches = [ training_data[ k : k + net.min_batch_size ] 
				for k in range(0, data_size , net.min_batch_size)]
			for min_batch in min_batches:
				net.__update_min_batch(min_batch, net.learning_rate)
			net.results_list.append(list(zip(net.biases, net.weights)))
			print(" Epoch" , '%2d' % (index+1) , "training complete")
		return(net.results_list)

	def evaluate(net, test_data):
		print('===== Evalutaion Started =====')
		sum_result = []
		data_size = len(test_data)
		for model in net.results_list:
			test_results = [(recognize(x, model), y) for (x, y) in test_data]
			actual_correct = sum(int(x == y) for (x, y) in test_results)
			sum_result.append(actual_correct)
			print(actual_correct, ' / ', data_size)
		return sum_result

#============================== Back Propagation ==============================#
	def __update_min_batch(net, min_batch, eta):
		nabla_b = [np.zeros(b.shape) for b in net.biases]
		nabla_w = [np.zeros(w.shape) for w in net.weights]
		for x, y in min_batch:
			delta_nabla_b, delta_nabla_w = net.__back_propagation(x, y)
			nabla_b = [nb+dnb for nb, dnb in list(zip(nabla_b, delta_nabla_b))]
			nabla_w = [nw+dnw for nw, dnw in list(zip(nabla_w, delta_nabla_w))]
		net.weights = [w-(eta/len(min_batch))*nw for w, nw in list(zip(net.weights, nabla_w))]
		net.biases = [b-(eta/len(min_batch))*nb for b, nb in list(zip(net.biases, nabla_b))]

	# Update weights and bias by measure of errors
	def __back_propagation(net, x, y):
		nabla_b = [np.zeros(b.shape) for b in net.biases]
		nabla_w = [np.zeros(w.shape) for w in net.weights]
		activation = x
		activations = [x] # list to store all the activations, layer by layer
		zs = [] # list to store all the z vectors, layer by layer
		for b, w in list(zip(net.biases, net.weights)):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		for l in range(2, net.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(net.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

#============================== Optional ==============================#
	def setParam(net, epoch, min_batch_size, learning_rate):
		net.epoch_amount   = epoch
		net.min_batch_size = min_batch_size
		net.learning_rate  = learning_rate

	def setModel(net, biaes, weights):
		net.biases  = biaes
		net.weights = weights

	def getModel(net):
		return(zip(net.biases, net.weights))

#============================== Miscellaneous functions ==============================#
# The sigmoid function.
def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

# Derivative of the sigmoid function.
def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

def cost_derivative(output_activations, y):
	return (output_activations-y)

# feedforward
def recognize(x ,model):
	for bias, weight in model: x = sigmoid(np.dot(weight, x) + bias)
	return np.argmax(x)

#============================== XIAOLI-20170318 ==============================#