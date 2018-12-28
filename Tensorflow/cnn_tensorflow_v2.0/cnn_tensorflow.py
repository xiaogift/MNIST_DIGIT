#
# ============================== Convolutional Neural Network use TensorFlow ============================== #
import tensorflow  as tf
import pickle_data as pickle

class ConvNets:

    def __init__(self):
        with tf.device('/cpu:0'):
            self.initialization()
            self.training()

    def initialization(self):
        image_x = tf.placeholder(tf.float32, [None, 784])
        label_y = tf.placeholder(tf.float32, [None, 10])
        kep_drp = tf.placeholder(tf.float32)
        x_image = tf.reshape(image_x, [-1,28,28,1])
        # -------------------------------------------------- #
        W_conv1 = self.weight_variable([5,5,1,32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.convolut_2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        # -------------------------------------------------- #
        W_conv2 = self.weight_variable([5,5,32,64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.convolut_2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        # -------------------------------------------------- #
        W_func1 = self.weight_variable([7*7*64, 1024])
        b_func1 = self.bias_variable([1024])
        h_polft = tf.reshape(h_pool2,[-1,7*7*64])
        h_func1 = tf.nn.relu(tf.matmul(h_polft, W_func1) + b_func1)
        h_fn1dp = tf.nn.dropout(h_func1, kep_drp)
        # -------------------------------------------------- #
        W_func2 = self.weight_variable([1024,10])
        b_func2 = self.bias_variable([10])
        predict = tf.nn.softmax(tf.matmul(h_fn1dp, W_func2) + b_func2)
        # -------------------------------------------------- #
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(label_y*tf.log(predict),reduction_indices=[1]))
        train_step    = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        # -------------------------------------------------- #
        self.train_step = train_step
        self.prediction = predict
        self.image_x    = image_x
        self.label_y    = label_y
        self.keep_drop  = kep_drp

    def testing(self, train_step, predict, image_x, label_y, kep_drp):
        result  = tf.cast(tf.argmax(predict, 1), tf.int64)
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        training_data, test_data = pickle.load_data()
        for index in range(3):
            for image_batch, label_batch in pickle.prepare_batch(data_source=training_data, capacity=5000, min_size=10, shuffle=True):
                session.run(train_step, feed_dict={image_x: image_batch, label_y: label_batch, kep_drp: 0.5})
            sample = session.run(result, feed_dict={test_data[0][1000]})
            print(sample, test_data[1][1000])

    def training(self):
        self.best_score = 0.0
        training_epoch  = 10
        data_capacity   = 5000
        data_min_size   = 10
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        training_data, test_data = pickle.load_data()
        for index in range(training_epoch):
            for image_batch, label_batch in pickle.prepare_batch(training_data, data_capacity, data_min_size, True):
                session.run(self.train_step, feed_dict={self.image_x: image_batch, self.label_y: label_batch, self.keep_drop: 0.5})
            current_score = self.compute_accuracy(session, test_data)
            if self.best_score < current_score:
                self.best_score = current_score
            print(current_score)
        print('Final score:', self.best_score)

    def compute_accuracy(self, session, test_data):
        label_z  = tf.placeholder(tf.int64, [None])
        prd_rlt  = session.run(self.prediction, feed_dict={self.image_x: test_data[0], self.keep_drop:1})
        correct  = tf.equal(tf.cast(tf.argmax(prd_rlt, 1), tf.int64), label_z)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        result   = session.run(accuracy, feed_dict={self.image_x: test_data[0], label_z: test_data[1], self.keep_drop:1})
        return result

    def weight_variable(self, shape): return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    def bias_variable(self,shape): return tf.Variable(tf.constant(0.1, shape=shape))
    def convolut_2d(self,x, W): return tf.nn.conv2d( x, W, strides=[1,1,1,1], padding='SAME')
    def max_pool_2x2(self,x): return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# ============================== Powered by XIAOLI 20170603 ============================== #
ConvNets()