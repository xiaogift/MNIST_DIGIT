#
# ============================== Convolutional Neural Network use TensorFlow ============================== #
import tensorflow  as tf
import pickle_data as pickle

class ConvNets:

    def initialization(self, features, labels, mode):
        input_layer = tf.reshape(features, [-1, 28, 28, 1])
        conv1_layer = tf.layers.conv2d(inputs=input_layer, filter=32, kernel_size=[5,5], padding='same', activation=tf.nn.relu)
        pool1_layer = tf.layers.max_pooling2d(inputs=conv1_layer, pool_size=[2,2], strides=2)
        conv2_layer = tf.layers.conv2d(inputs=pool1_layer, filter=64, kernel_size=[5,5], padding='same', activation=tf.nn.relu)
        pool2_layer = tf.layers.max_pooling2d(inputs=conv2_layer, pool_size=[2,2], strides=2)
        pflat_layer = tf.reshape(pool2_layer, [-1, 7*7*64])
        dense_layer = tf.layers.dense(inputs=pflat_layer,units=1024, activation=tf.nn.relu)
        drput_layer = tf.layers.dropout(inputs=dense_layer, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
        logts_layer = tf.layers.dense(inputs=drput_layer, units=10)

        loss = None
        train_op = None

        if mode != learn_ModeKeys.INFER:
            onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logts_layer)

        if mode == learn_ModeKeys.INFER:
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss, 
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.001,
                optimizer='SGD')

        predictions = {
            'class': tf.argmax(input=logts_layer, axis=1),
            'probabilities': tf.nn.softmax(logts_layer, name='softmax_tensor') }

        return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)

# ============================== Powered by XIAOLI 20170605 ============================== #
