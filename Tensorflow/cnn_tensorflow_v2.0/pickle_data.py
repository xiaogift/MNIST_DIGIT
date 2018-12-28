from PIL import Image
import pickle, gzip, numpy as np

file_location = '../MNIST_data/mnist.pkl.gz'
#================================= MINIST DATA =================================#
def load_data(validation=False):
    f = gzip.open( file_location , 'rb')
    raw_train, raw_validation, raw_test = pickle.load(f, encoding='iso-8859-1')
    f.close()
    if not validation: return list(zip(raw_train[0], [vectorized_label(y) for y in raw_train[1]])), raw_test
    return training_data, raw_validation, raw_test

def prepare_batch(data_source, capacity, min_size, shuffle=False):
    if shuffle: np.random.shuffle(data_source)
    min_batches = [data_source[k:k+min_size] for k in range(0, capacity , min_size)]
    norm_image  = []
    norm_label  = []
    for min_batch in min_batches:
        image_batch = []
        label_batch = []
        for x,y in min_batch:
            image_batch.append(x)
            label_batch.append(y)
        norm_image.append(image_batch)
        norm_label.append(label_batch)
    return list(zip(norm_image, norm_label))

def vectorized_label(j):
    e    = np.zeros((10))
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
    array_2digit(test_data[index][0]).show()

#============================== XIAOLI-20170318 ==============================#