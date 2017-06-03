import tensorflow as tf

x = tf.constant(107.13, name='input')
w = tf.Variable(0.75, name='weight')
y = tf.multiply(w, x, name='output')
y_ = tf.constant(17.23, name='correct_value')
loss = tf.pow(y - y_, 2, name='loss')
train_step = tf.train.GradientDescentOptimizer(13.27).minimize(loss)

for value in [x, w, y, y_, loss]:
     tf.summary.scalar(value.op.name, value)

# log / graph
summaries = tf.summary.merge_all()

# start session & logger
sess = tf.Session()
summary_writer = tf.summary.FileWriter('log_simple_stats', sess.graph)

sess.run(tf.global_variables_initializer())
for i in range(10000):
    summary_writer.add_summary(sess.run(summaries), i)
    sess.run(train_step)