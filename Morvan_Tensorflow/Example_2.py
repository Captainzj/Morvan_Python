import tensorflow as tf
import numpy as np
'''
简单输出train的阶段性结果
'''
# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

## create Tensorflow Structure start##
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases 

loss = tf.reduce_mean(tf.square(y-y_data)) 
optimizer = tf.train.GradientDescentOptimizer(0.5)  # learning rate: 0.5
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()  # init the variables
## create Tensorflow Structure end##

sess = tf.Session()
sess.run(init)          # Very important  指向处理位置  激活init

for step in range(201):
    sess.run(train)  # 激活train
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))