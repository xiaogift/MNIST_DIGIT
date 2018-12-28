#
# ============================== Convolutional Neural Network use TensorFlow ============================== #
import tensorflow  as tf
import pickle_data as pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np

class ConvNets:

    def __init__(self):
        self.training_data, self.test_data = pickle.load_data()
        with tf.device('/cpu:0'): self.build_network()

    def build_network(self):
        image_x = tf.placeholder(tf.float32, [None, 784], name="x")
        label_y = tf.placeholder(tf.float32, [None,  10], name="labels")
        kep_drp = tf.placeholder(tf.float32, name="dropout")
        x_image = tf.reshape(image_x, [-1,28,28,1], name="image")
        tf.summary.image("input", x_image, 3)
        # -------------------------------------------------- #
        layer1 = self.build_conv_layer("conv1", x_image, [5,5,1,32], [32])
        layer2 = self.build_conv_layer("conv2", layer1, [5,5,32,64], [64])
        # -------------------------------------------------- #
        with tf.name_scope("fc1"):
            W_func1 = self.weight_variable([7*7*64, 1024])
            b_func1 = self.bias_variable([1024])
            layer3  = tf.reshape(layer2, [-1,7*7*64])
            h_func1 = tf.nn.relu(tf.matmul(layer3, W_func1) + b_func1)
            # h_fn1dp = tf.nn.dropout(h_func1, kep_drp)
        # -------------------------------------------------- #
        with tf.name_scope("fc2"):
            W_func2 = self.weight_variable([1024,10])
            b_func2 = self.bias_variable([10])
            predict = tf.nn.softmax(tf.matmul(h_func1, W_func2) + b_func2)
        # -------------------------------------------------- #
        with tf.name_scope("xent"):
            xent = tf.reduce_mean(-tf.reduce_sum(label_y*tf.log(predict),reduction_indices=[1]))
        tf.summary.scalar("cross_entropy",xent)
        # -------------------------------------------------- #
        with tf.name_scope("train"):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(xent)

        best_score     = 0.0
        training_epoch = 30
        data_capacity  = 5000
        data_min_size  = 10
        LOG_DIR        = "./mnist_demo/tensorboard"
        session        = tf.Session()
        merged_summary = tf.summary.merge_all()
        writer         = tf.summary.FileWriter(LOG_DIR)
        session.run(tf.global_variables_initializer())
        writer.add_graph(session.graph)

        embedding  = tf.Variable(tf.zeros([1024, 1024]), name="test_embedding")
        assignment = embedding.assign(h_func1)
        LABELS     = os.path.join(os.getcwd(), "labels_1024.tsv")
        SPRITES    = os.path.join(os.getcwd(), "sprite_1024.png")
        config     = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        embedding_config                   = config.embeddings.add()
        embedding_config.tensor_name       = embedding.name
        embedding_config.sprite.image_path = SPRITES
        embedding_config.metadata_path     = LABELS
        embedding_config.sprite.single_image_dim.extend([28, 28])
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
        mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir="../../MNIST_DATA", one_hot=True)
        saver = tf.train.Saver()

        for index in range(training_epoch):
            for image_batch, label_batch in pickle.prepare_batch(self.training_data, data_capacity, data_min_size, True):
                session.run(train_step, feed_dict={image_x: image_batch, label_y: label_batch, kep_drp: 0.5})
            current_score = self.compute_accuracy(self.test_data, session, image_x, kep_drp, predict)
            best_score    = self.score_board(current_score, best_score)
            summary       = session.run(merged_summary, feed_dict={image_x: image_batch, label_y: label_batch, kep_drp: 0.5})
            tf.summary.scalar("current_score",current_score)
            session.run(assignment, feed_dict={image_x: mnist.test.images[:1024], label_y: mnist.test.labels[:1024]})
            saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), index)
            writer.add_summary(summary, index)
        print('The best result:', best_score)

    # def sample(self, train_step, predict, image_x, label_y, kep_drp):
    #     result  = tf.cast(tf.argmax(predict, 1), tf.int64)
    #     session = tf.Session()
    #     session.run(tf.global_variables_initializer())
    #     for _ in range(3):
    #         for image_batch, label_batch in pickle.prepare_batch(data_source=self.training_data, capacity=5000, min_size=10, shuffle=True):
    #             session.run(train_step, feed_dict={image_x: image_batch, label_y: label_batch, kep_drp: 0.5})
    #         sample = session.run(result, feed_dict={self.test_data[0][1000]})
    #         print(sample, self.test_data[1][1000])

    def compute_accuracy(self, test_data, session, image_x, keep_drop, prediction):
        label_z   = tf.placeholder(tf.int64, [None])
        prd_rlt   = session.run(prediction, feed_dict={image_x: test_data[0], keep_drop:1})
        print(np.shape(prd_rlt))
        print(prd_rlt)
        correct   = tf.equal(tf.cast(tf.argmax(prd_rlt, 1), tf.int64), label_z)
        accuracy  = tf.reduce_mean(tf.cast(correct, tf.float32))
        return session.run(accuracy, feed_dict={image_x: test_data[0], label_z: test_data[1], keep_drop:1})

    def build_conv_layer(self, layer_name, input_data, p_weight, p_bias):
        with tf.name_scope(layer_name):
            W_conv = self.weight_variable(p_weight)
            b_conv = self.bias_variable(p_bias)
            h_conv = tf.nn.relu(self.convolut_2d(input_data, W_conv) + b_conv)
            output = self.max_pool_2x2(h_conv)
            tf.summary.histogram('weights',W_conv)
            tf.summary.histogram('bias',b_conv)
            tf.summary.histogram('activations',h_conv)
            return output

    def score_board(self, current_score, best_score):
        print("Current epoch accuracy is",current_score)
        if best_score < current_score: return current_score
        return best_score

    def weight_variable(self, shape): return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="W")
    def bias_variable(self,shape): return tf.Variable(tf.constant(0.1, shape=shape), name="B")
    def convolut_2d(self,x, W): return tf.nn.conv2d( x, W, strides=[1,1,1,1], padding="SAME")
    def max_pool_2x2(self,x): return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

# ============================== Powered by XIAOLI 20170603 ============================== #
ConvNets()

# Loaded MNIST data: 50000 for training and 10000 for testing.
# Extracting ../../MNIST_DATA\train-images-idx3-ubyte.gz
# Extracting ../../MNIST_DATA\train-labels-idx1-ubyte.gz
# Extracting ../../MNIST_DATA\t10k-images-idx3-ubyte.gz
# Extracting ../../MNIST_DATA\t10k-labels-idx1-ubyte.gz
# Current epoch accuracy is 0.9101
# Current epoch accuracy is 0.9225
# Current epoch accuracy is 0.9345
# Current epoch accuracy is 0.9631
# Current epoch accuracy is 0.9571
# Current epoch accuracy is 0.97
# Current epoch accuracy is 0.9718
# Current epoch accuracy is 0.9725
# Current epoch accuracy is 0.9795
# Current epoch accuracy is 0.9715
# Current epoch accuracy is 0.9719
# Current epoch accuracy is 0.9789
# Current epoch accuracy is 0.9804
# Current epoch accuracy is 0.9835
# Current epoch accuracy is 0.981
# Current epoch accuracy is 0.9814
# Current epoch accuracy is 0.9816
# Current epoch accuracy is 0.9799
# Current epoch accuracy is 0.9847
# Current epoch accuracy is 0.9855
# Current epoch accuracy is 0.9873
# Current epoch accuracy is 0.9854
# Current epoch accuracy is 0.9813
# Current epoch accuracy is 0.9864
# Current epoch accuracy is 0.9855
# Current epoch accuracy is 0.986
# Current epoch accuracy is 0.9885
# Current epoch accuracy is 0.9882
# Current epoch accuracy is 0.9871
# Current epoch accuracy is 0.9875
# The best result: 0.9885
# [Finished in 1499.7s]