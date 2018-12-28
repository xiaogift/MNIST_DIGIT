import mnist_loader
import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
net.SGD(list(training_data), 30, 10, 0.1, lmbda=5.0, evaluation_data=list(validation_data), monitor_evaluation_accuracy=True)

"""
# net.large_weight_initializer()
Epoch 0 training complete
Accuracy on evaluation data: 8774 / 10000
Epoch 1 training complete
Accuracy on evaluation data: 9052 / 10000
Epoch 2 training complete
Accuracy on evaluation data: 9178 / 10000
Epoch 3 training complete
Accuracy on evaluation data: 9265 / 10000
Epoch 4 training complete
Accuracy on evaluation data: 9324 / 10000
Epoch 5 training complete
Accuracy on evaluation data: 9398 / 10000
Epoch 6 training complete
Accuracy on evaluation data: 9437 / 10000
Epoch 7 training complete
Accuracy on evaluation data: 9465 / 10000
Epoch 8 training complete
Accuracy on evaluation data: 9458 / 10000
Epoch 9 training complete
Accuracy on evaluation data: 9493 / 10000
Epoch 10 training complete
Accuracy on evaluation data: 9519 / 10000
Epoch 11 training complete
Accuracy on evaluation data: 9540 / 10000
Epoch 12 training complete
Accuracy on evaluation data: 9534 / 10000
Epoch 13 training complete
Accuracy on evaluation data: 9567 / 10000
Epoch 14 training complete
Accuracy on evaluation data: 9547 / 10000
Epoch 15 training complete
Accuracy on evaluation data: 9559 / 10000
Epoch 16 training complete
Accuracy on evaluation data: 9570 / 10000
Epoch 17 training complete
Accuracy on evaluation data: 9593 / 10000
Epoch 18 training complete
Accuracy on evaluation data: 9614 / 10000
Epoch 19 training complete
Accuracy on evaluation data: 9602 / 10000
Epoch 20 training complete
Accuracy on evaluation data: 9611 / 10000
Epoch 21 training complete
Accuracy on evaluation data: 9619 / 10000
Epoch 22 training complete
Accuracy on evaluation data: 9612 / 10000
Epoch 23 training complete
Accuracy on evaluation data: 9599 / 10000
Epoch 24 training complete
Accuracy on evaluation data: 9629 / 10000
Epoch 25 training complete
Accuracy on evaluation data: 9602 / 10000
Epoch 26 training complete
Accuracy on evaluation data: 9629 / 10000
Epoch 27 training complete
Accuracy on evaluation data: 9603 / 10000
Epoch 28 training complete
Accuracy on evaluation data: 9640 / 10000
Epoch 29 training complete
Accuracy on evaluation data: 9610 / 10000
[Finished in 248.7s]
"""
"""
Epoch 0 training complete
Accuracy on evaluation data: 9267 / 10000
Epoch 1 training complete
Accuracy on evaluation data: 9401 / 10000
Epoch 2 training complete
Accuracy on evaluation data: 9433 / 10000
Epoch 3 training complete
Accuracy on evaluation data: 9507 / 10000
Epoch 4 training complete
Accuracy on evaluation data: 9530 / 10000
Epoch 5 training complete
Accuracy on evaluation data: 9549 / 10000
Epoch 6 training complete
Accuracy on evaluation data: 9551 / 10000
Epoch 7 training complete
Accuracy on evaluation data: 9571 / 10000
Epoch 8 training complete
Accuracy on evaluation data: 9593 / 10000
Epoch 9 training complete
Accuracy on evaluation data: 9583 / 10000
Epoch 10 training complete
Accuracy on evaluation data: 9581 / 10000
Epoch 11 training complete
Accuracy on evaluation data: 9597 / 10000
Epoch 12 training complete
Accuracy on evaluation data: 9603 / 10000
Epoch 13 training complete
Accuracy on evaluation data: 9603 / 10000
Epoch 14 training complete
Accuracy on evaluation data: 9610 / 10000
Epoch 15 training complete
Accuracy on evaluation data: 9587 / 10000
Epoch 16 training complete
Accuracy on evaluation data: 9605 / 10000
Epoch 17 training complete
Accuracy on evaluation data: 9597 / 10000
Epoch 18 training complete
Accuracy on evaluation data: 9614 / 10000
Epoch 19 training complete
Accuracy on evaluation data: 9612 / 10000
Epoch 20 training complete
Accuracy on evaluation data: 9600 / 10000
Epoch 21 training complete
Accuracy on evaluation data: 9622 / 10000
Epoch 22 training complete
Accuracy on evaluation data: 9617 / 10000
Epoch 23 training complete
Accuracy on evaluation data: 9613 / 10000
Epoch 24 training complete
Accuracy on evaluation data: 9625 / 10000
Epoch 25 training complete
Accuracy on evaluation data: 9629 / 10000
Epoch 26 training complete
Accuracy on evaluation data: 9616 / 10000
Epoch 27 training complete
Accuracy on evaluation data: 9625 / 10000
Epoch 28 training complete
Accuracy on evaluation data: 9631 / 10000
Epoch 29 training complete
Accuracy on evaluation data: 9624 / 10000
[Finished in 253.4s]
"""