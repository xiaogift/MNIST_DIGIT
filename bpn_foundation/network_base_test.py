import matplotlib.pylab as plot
import network_base     as net
import pickle_data      as pickle

#============================== Decoupling ==============================#
def training( module_file, training_data, save = False ):
    __model = net.Network( [ 784 , 30 , 10 ] )
    __model.setParam( epoch=50, min_batch_size=10, learning_rate=1.5 )
    __model.training(training_data)
    if save: pickle.save_network( __model , module_file )

def evaluate( models, data, trend = False):
    if trend:
        plot.plot(list(range(50)), models.evaluate(data))
        plot.show()
    else: models.evaluate(data)

def sample( model, data, index, show = True ):
    if show: pickle.show_pic(index)
    result = net.recognize(data[index][0], model)
    print(' # Original: {} Prediction: {}'.format(data[index][1], result))

def test(model, data):
    result = net.recognize(data, model)
    print('I guess it is: ',result)
    return result

#============================== XIAOLI-20170318 ==============================#
# module_file = '../MODEL_DATA/decent_network.plk'
# training_data, validation_data, test_data = pickle.load_data_wrapper()
# # training(module_file, training_data, True)
# evaluate(pickle.load_network(module_file), test_data, True)

