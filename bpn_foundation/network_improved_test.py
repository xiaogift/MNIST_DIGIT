import network_improved as network
import pickle_data      as pickle

training_data, validation_data, test_data = pickle.load_data_wrapper()
net = network.Network([784, 30, 10], cost=network.CrossEntropyCost)
net.SGD(list(training_data), 30, 10, 0.1, lmbda=5.0, evaluation_data=list(validation_data), monitor_evaluation_accuracy=True)

"""
Epoch 0 training complete
Accuracy on evaluation data: 9290 / 10000
Epoch 1 training complete
Accuracy on evaluation data: 9393 / 10000
Epoch 2 training complete
Accuracy on evaluation data: 9482 / 10000
Epoch 3 training complete
Accuracy on evaluation data: 9497 / 10000
Epoch 4 training complete
Accuracy on evaluation data: 9542 / 10000
Epoch 5 training complete
Accuracy on evaluation data: 9516 / 10000
Epoch 6 training complete
Accuracy on evaluation data: 9586 / 10000
Epoch 7 training complete
Accuracy on evaluation data: 9586 / 10000
Epoch 8 training complete
Accuracy on evaluation data: 9600 / 10000
Epoch 9 training complete
Accuracy on evaluation data: 9592 / 10000
Epoch 10 training complete
Accuracy on evaluation data: 9605 / 10000
Epoch 11 training complete
Accuracy on evaluation data: 9617 / 10000
Epoch 12 training complete
Accuracy on evaluation data: 9616 / 10000
Epoch 13 training complete
Accuracy on evaluation data: 9591 / 10000
Epoch 14 training complete
Accuracy on evaluation data: 9610 / 10000
Epoch 15 training complete
Accuracy on evaluation data: 9631 / 10000
Epoch 16 training complete
Accuracy on evaluation data: 9597 / 10000
Epoch 17 training complete
Accuracy on evaluation data: 9629 / 10000
Epoch 18 training complete
Accuracy on evaluation data: 9618 / 10000
Epoch 19 training complete
Accuracy on evaluation data: 9622 / 10000
Epoch 20 training complete
Accuracy on evaluation data: 9614 / 10000
Epoch 21 training complete
Accuracy on evaluation data: 9649 / 10000
Epoch 22 training complete
Accuracy on evaluation data: 9642 / 10000
Epoch 23 training complete
Accuracy on evaluation data: 9623 / 10000
Epoch 24 training complete
Accuracy on evaluation data: 9627 / 10000
Epoch 25 training complete
Accuracy on evaluation data: 9638 / 10000
Epoch 26 training complete
Accuracy on evaluation data: 9629 / 10000
Epoch 27 training complete
Accuracy on evaluation data: 9621 / 10000
Epoch 28 training complete
Accuracy on evaluation data: 9640 / 10000
Epoch 29 training complete
Accuracy on evaluation data: 9627 / 10000
[Finished in 275.9s]
"""