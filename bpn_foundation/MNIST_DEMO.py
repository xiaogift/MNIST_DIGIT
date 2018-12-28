#
#============================== GO ==============================#
import network_base_test  as demo
import numpy, pickle_data as pickle
import cv2

def pic_2data(pic):
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic = cv2.resize(pic, (28,28))
    data = numpy.asarray(pic)
    return numpy.reshape(data, [784])

module_file = '../MODEL_DATA/decent_network.plk'
training_data, validation_data, test_data = pickle.load_data_wrapper()
model = pickle.load_network(module_file)
# demo.evaluate(model, test_data, True)
for _ in range (9):
    index = int(numpy.random.randint(1,1000,1))
    demo.sample(model.results_list[-1], test_data, index)

# Video demo
# for i in range(1000):
#     v, frame = cv2.VideoCapture(0).read()
#     cv2.imshow('XIAOLI', frame)
#     demo.test(model.results_list[-1], pic_2data(frame))
#     cv2.waitKey(1)

#============================== XIAOLI-20170608 ==============================#