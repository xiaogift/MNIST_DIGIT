import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as data
mnist = data.read_data_sets('MNIST_data', one_hot=True)

# Extraction #
def weight_variable(shape): return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
def bias_variable(shape): return tf.Variable(tf.constant(0.1, shape=shape))
def convolut_2d(x, W): return tf.nn.conv2d( x, W, strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x): return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# Caculation #
def compute_accuracy(image_data, label_data):
    global predict
    prd_rlt  = sess.run(predict, feed_dict={image_x: image_data, kep_drp:1})
    correct  = tf.equal(tf.argmax(prd_rlt, 1),tf.argmax(label_data, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    result   = sess.run(accuracy, feed_dict={image_x: image_data, label_y: label_data, kep_drp:1})
    return result

# Definition #
image_x = tf.placeholder(tf.float32, [None, 784])
label_y = tf.placeholder(tf.float32, [None, 10])
kep_drp = tf.placeholder(tf.float32)
x_image = tf.reshape(image_x, [-1,28,28,1])
# Buld convolutional layer 1 #
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(convolut_2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# Buld convolutional layer 2 #
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(convolut_2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# Buld full connected layer 1 #
W_func1 = weight_variable([7*7*64, 1024])
b_func1 = bias_variable([1024])
h_polft = tf.reshape(h_pool2,[-1,7*7*64])
h_func1 = tf.nn.relu(tf.matmul(h_polft, W_func1) + b_func1)
h_fn1dp = tf.nn.dropout(h_func1, kep_drp)
# Buld full connected layer 2 #
W_func2 = weight_variable([1024,10])
b_func2 = bias_variable([10])
predict = tf.nn.softmax(tf.matmul(h_fn1dp, W_func2) + b_func2)
# Prepare to run #
cross_entropy = tf.reduce_mean(-tf.reduce_sum(label_y*tf.log(predict),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# Evaluation #
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={image_x: batch_xs, label_y: batch_ys, kep_drp: 0.5})
    if i % 50 == 0:
        print(compute_accuracy( mnist.test.images, mnist.test.labels ))

# Extracting MNIST_data\train-images-idx3-ubyte.gz
# Extracting MNIST_data\train-labels-idx1-ubyte.gz
# Extracting MNIST_data\t10k-images-idx3-ubyte.gz
# Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
# 2017-05-30 10:01:03.096282: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE instructions, but these are available on your machine and could speed up CPU computations.
# 2017-05-30 10:01:03.096282: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE2 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-05-30 10:01:03.096282: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-05-30 10:01:03.096282: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-05-30 10:01:03.096282: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-05-30 10:01:03.096282: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
# 2017-05-30 10:01:03.111882: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-05-30 10:01:03.111882: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
# 0.1311
# 0.7833
# 0.8835
# 0.915
# 0.9283
# 0.94
# 0.9439
# 0.9508
# 0.9565
# 0.9604
# 0.9623
# 0.9634
# 0.9639
# 0.9665
# 0.9669
# 0.9689
# 0.9695
# 0.9693
# 0.9698
# 0.9735
# [Finished in 689.8s]