import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
'''
可视化train的过程
'''
def add_layer(inputs,in_size,out_size,activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs,Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs

# import data
x_data = np.linspace(-1,1,300,dtype = np.float32)[:,np.newaxis]  # 300 examples, 1 feature
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

# add hidden layer
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)

# add output layer
prediction = add_layer(l1,10,1,activation_function=None)

# the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 优化,提升准确率


sess = tf.Session()
# important step
sess.run(tf.global_variables_initializer())

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion() #本次运行请注释，全局运行不要注释 # 用于连续显示
plt.show()


for i in range(1000):
    # training
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})  #Optimizer
    if i % 50 == 0:
        # to see the step improvement (loss is smaller and smaller)
        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))

        #to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        #plot the prediction
        lines = ax.plot(x_data,prediction_value,'r-',lw = 5)
        plt.pause(0.1)
