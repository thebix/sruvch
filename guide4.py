# https://www.tensorflow.org/get_started/mnist/beginners
import tensorflow as tf

# download data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#
# Implementing the Regression
#

# 28x28 matrix = 784 numbers input
INIT_X_LENGTH = 784
# 10x1 matrix (array) = 10 numbers output
INIT_Y_LENGTH = 10

#The MNIST data is split into three parts: 55,000 data points of training data (mnist.train), 
# 10,000 points of test data (mnist.test), and 5,000 points of validation data (mnist.validation).
TRAIN_DATA_COUNT = 55000 # number of train images 28x28
TEST_DATA_COUNT = 10000 # ? number of test images 28x28 ?
VALIDATE_DATA_COUNT = 5000 # ? number of validate images 28x28 ?

TRAIN_COUNT = 1000
TRAIN_BATCH_LENGTH = 100
TRAIN_OPTIMIZER = 0.5

# mnist.train.images is a tensor wih shape [55000, 784]
# The first dimension is an index into the list of images and the second dimension is the index for each pixel in each image.
# Each entry in the tensor is a pixel intensity between 0 and 1, for a particular pixel in a particular image.

# Each image in MNIST has a corresponding label, a number between 0 and 9 representing the digit drawn in the image.
# A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension.
# mnist.train.labels is a [55000, 10] array of floats.

#
# Create the model
#

# None - any number of data, INIT_X_LENGTH - dimensional vector of input 
x = tf.placeholder(tf.float32, [None, INIT_X_LENGTH]) # Here None means that a dimension can be of any length
W = tf.Variable(tf.zeros([INIT_X_LENGTH, INIT_Y_LENGTH])) # input length -> output length
b = tf.Variable(tf.zeros([INIT_Y_LENGTH])) # output length
y = tf.matmul(x, W) + b
# Since we are going to learn W and b, it doesn't matter very much what they initially are.


#
# Training
#
# correct answers
y_ = tf.placeholder(tf.float32, [None, INIT_Y_LENGTH])

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# can be numerically unstable.
#
# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
# outputs of 'y', and then average across the batch.
cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(TRAIN_OPTIMIZER).minimize(cross_entropy)
# In this case, we ask TensorFlow to minimize cross_entropy using the gradient descent algorithm with a learning 
# rate of 0.5. Gradient descent is a simple procedure, where TensorFlow simply shifts each variable a little bit 
# in the direction that reduces the cost.

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(TRAIN_COUNT):
    batch_xs, batch_ys = mnist.train.next_batch(TRAIN_BATCH_LENGTH) # get batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# Each step of the loop, we get a "batch" of one hundred random data points from our training set. 
# We run train_step feeding in the batches data to replace the placeholders.
# Using small batches of random data is called stochastic training -- in this case, stochastic gradient descent. 
# Ideally, we'd like to use all our data for every step of training, but that's expensive. 
# So, instead, we use a different subset every time. Doing this is cheap and has much of the same benefit.

# tf.argmax(y, 1) - index of highest possibility label. example: [0,1...,0] - letter 1
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # indexes must be the same
# That gives us a list of booleans. To determine what fraction are correct, we cast to floating point numbers 
# and then take the mean. For example, [True, False, True, True] would become [1,0,1,1] which would become 0.75.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))