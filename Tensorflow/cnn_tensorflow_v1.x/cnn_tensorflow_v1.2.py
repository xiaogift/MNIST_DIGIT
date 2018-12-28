#
# ============================== Convolutional Neural Network use TensorFlow ============================== #
import tensorflow as tf

class Convnets:

    def __init__(self):
        self.network_build()

    def network_build(self):
        # Definition #
        image_x = tf.placeholder(tf.float32, [None, 784])
        label_y = tf.placeholder(tf.float32, [None, 10])
        kep_drp = tf.placeholder(tf.float32)
        x_image = tf.reshape(image_x, [-1,28,28,1])
        # Buld convolutional layer 1 #
        W_conv1 = self.weight_variable([5,5,1,32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.convolut_2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        # Buld convolutional layer 2 #
        W_conv2 = self.weight_variable([5,5,32,64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.convolut_2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        # Buld full connected layer 1 #
        W_func1 = self.weight_variable([7*7*64, 1024])
        b_func1 = self.bias_variable([1024])
        h_polft = tf.reshape(h_pool2,[-1,7*7*64])
        h_func1 = tf.nn.relu(tf.matmul(h_polft, W_func1) + b_func1)
        h_fn1dp = tf.nn.dropout(h_func1, kep_drp)
        # Buld full connected layer 2 #
        W_func2 = self.weight_variable([1024,10])
        b_func2 = self.bias_variable([10])
        predict = tf.nn.softmax(tf.matmul(h_fn1dp, W_func2) + b_func2)
        # Prepare to train #
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(label_y*tf.log(predict),reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        
        self.evaluation( train_step, predict, image_x, label_y, kep_drp )

    # Evaluation #
    def evaluation(self, train_step, predict, image_x, label_y, kep_drp):
        sess  = tf.Session()
        sess.run(tf.global_variables_initializer())
        mnist = self.load_data()
        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={image_x: batch_xs, label_y: batch_ys, kep_drp: 0.5})
            if i % 50 == 0:
                print(self.compute_accuracy( sess, predict,image_x, mnist.test.images, label_y, mnist.test.labels, kep_drp ))
    # Caculation #
    def compute_accuracy(self, session, prediction, image_x, image_data, label_y, label_data, keep_prob):
        prd_rlt  = session.run(prediction, feed_dict={image_x: image_data, keep_prob:1})
        correct  = tf.equal(tf.argmax(prd_rlt, 1),tf.argmax(label_data, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        result   = session.run(accuracy, feed_dict={image_x: image_data, label_y: label_data, keep_prob:1})
        return result
    # Extraction #
    def load_data(self):
        import tensorflow.examples.tutorials.mnist.input_data as mnist
        return mnist.read_data_sets('MNIST_data', one_hot=True)
    def weight_variable(self, shape): return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    def bias_variable(self,shape): return tf.Variable(tf.constant(0.1, shape=shape))
    def convolut_2d(self,x, W): return tf.nn.conv2d( x, W, strides=[1,1,1,1], padding='SAME')
    def max_pool_2x2(self,x): return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# ============================== Powered by XIAOLI 20170530 ============================== #
Convnets()
# Extracting MNIST_data\train-images-idx3-ubyte.gz
# Extracting MNIST_data\train-labels-idx1-ubyte.gz
# Extracting MNIST_data\t10k-images-idx3-ubyte.gz
# Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
# 2017-05-30 10:23:29.267046: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE instructions, but these are available on your machine and could speed up CPU computations.
# 2017-05-30 10:23:29.267046: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE2 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-05-30 10:23:29.267046: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-05-30 10:23:29.267046: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-05-30 10:23:29.267046: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-05-30 10:23:29.267046: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
# 2017-05-30 10:23:29.267046: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-05-30 10:23:29.267046: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
# 0.1179
# 0.7608
# 0.8746
# 0.9071
# 0.9161
# 0.9311
# 0.9363
# 0.942
# 0.9489
# 0.949
# 0.9527
# 0.9522
# 0.958
# 0.9599
# 0.9599
# 0.963
# 0.9644
# 0.9631
# 0.9673
# 0.9687
# [Finished in 688.1s]

# 2017-06-01 07:21:04.985849: W c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE instructions, but these are available on your machine and could speed up CPU computations.
# 2017-06-01 07:21:04.986849: W c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE2 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-06-01 07:21:04.986849: W c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-06-01 07:21:04.987849: W c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-06-01 07:21:04.987849: W c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-06-01 07:21:04.987849: W c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
# 2017-06-01 07:21:04.988849: W c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-06-01 07:21:04.988849: W c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
# 2017-06-01 07:21:05.647915: I c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\common_runtime\gpu\gpu_device.cc:887] Found device 0 with properties: 
# name: Quadro M5000
# major: 5 minor: 2 memoryClockRate (GHz) 1.038
# pciBusID 0000:02:00.0
# Total memory: 8.00GiB
# Free memory: 7.79GiB
# 2017-06-01 07:21:05.648915: I c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\common_runtime\gpu\gpu_device.cc:908] DMA: 0 
# 2017-06-01 07:21:05.648915: I c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\common_runtime\gpu\gpu_device.cc:918] 0:   Y 
# 2017-06-01 07:21:05.648915: I c:\tf_jenkins\home\workspace\nightly-win\device\gpu\os\windows\tensorflow\core\common_runtime\gpu\gpu_device.cc:977] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Quadro M5000, pci bus id: 0000:02:00.0)
# Extracting MNIST_data\train-images-idx3-ubyte.gz
# Extracting MNIST_data\train-labels-idx1-ubyte.gz
# Extracting MNIST_data\t10k-images-idx3-ubyte.gz
# Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
# 0.1282
# 0.813
# 0.877
# 0.904
# 0.9212
# 0.9307
# 0.9382
# 0.9434
# 0.9469
# 0.9487
# 0.9518
# 0.9528
# 0.958
# 0.9586
# 0.9631
# 0.963
# 0.9633
# 0.9679
# 0.9671
# 0.9691
# [Finished in 30.1s]