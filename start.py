import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

print("\n\nStart...")
PATH_STATS = 'storage/stats'
INIT_W = .8
INIT_X = .3
INIT_B = -.3
TRAIN_VALUES = .0
TRAIN_OPTIMIZER = 0.05
TRAIN_COUNT = 100

# Model parameters
w = tf.Variable(INIT_W, name='weight')
b = tf.Variable(INIT_B, tf.float32)

# Model input / output / test
x = tf.constant(INIT_X, name='input')
multi = tf.multiply(w, x)
y = tf.add(multi, b, name="output")
y_ = tf.constant(TRAIN_VALUES, name='correct_value')

# loss
# loss = tf.pow(y - y_, 2, name='loss')
loss = tf.reduce_sum(tf.square(y - y_)) # sum of the squares

# optimizer
optimizer = tf.train.GradientDescentOptimizer(TRAIN_OPTIMIZER)
train = optimizer.minimize(loss)

# log / graph
for value in [x, w, y, y_, loss]:
     tf.summary.scalar(value.op.name, value)
summaries = tf.summary.merge_all()

# start session & logger
sess = tf.Session()
summary_writer = tf.summary.FileWriter(PATH_STATS, sess.graph)

sess.run(tf.global_variables_initializer())
for i in range(TRAIN_COUNT):
    summary_writer.add_summary(sess.run(summaries), i)
    sess.run(train)