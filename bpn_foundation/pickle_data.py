from PIL import Image
import pickle, gzip, numpy as np

file_location = '../MNIST_DATA/mnist.pkl.gz'
#================================= MINIST DATA =================================#
def load_data():
    f = gzip.open( file_location , 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='iso-8859-1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d  = load_data()
    training_inputs   = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results  = [vectorized_result(y) for y in tr_d[1]]
    training_data     = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data   = list(zip(validation_inputs, va_d[1]))
    test_inputs       = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data         = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e    = np.zeros((10, 1))
    e[j] = 1.0
    return e

#================================= MODEL OPERATION =================================#
def save_network( network , network_path ):
    file = open( network_path ,"wb" )
    pickle.dump(network, file)
    file.close()
    print('===== MODELS UPDATED =====')

def load_network( network_path ):
    file = open(network_path,"rb")
    network_obj  = pickle.load(file)
    file.close()
    return network_obj

#================================= PICTURE OPERATION =================================#
def array_2digit(array_data):
    return Image.fromarray(np.array(np.dot(array_data, 255), dtype='uint8').reshape(28,28))

def show_pic(index):
    training_data, validation_data, test_data = load_data_wrapper()
    image = array_2digit(test_data[index][0])
    image.resize((100,100), Image.LANCZOS).show()

#============================== XIAOLI-20170318 ==============================#