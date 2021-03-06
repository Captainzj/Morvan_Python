import tensorflow as tf
import numpy as np

## Save to file
# remember to define the same dtype and shape when restore
W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, "my_net/save_net.ckpt")
    print("Save to path: ", save_path)

"""
Save to path:  my_net/save_net.ckpt
"""

# ###提取###
# # 先建立 W, b 的容器
# W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
# b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

# # 这里不需要初始化步骤 init= tf.initialize_all_variables()

# saver = tf.train.Saver()
# with tf.Session() as sess:
#     # 提取变量
#     saver.restore(sess, "my_net/save_net.ckpt")
#     print("weights:", sess.run(W))
#     print("biases:", sess.run(b))

"""
weights: [[ 1.  2.  3.]
          [ 3.  4.  5.]]
biases: [[ 1.  2.  3.]]
"""