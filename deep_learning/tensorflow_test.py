import tensorflow as tf
# declare two symbolic floating-point scalar
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
# create asymbolic expression using add function
add = tf.add(a,b)
# bind 1.5 to a, 2.5 to b, and evaluate c
sess = tf.Session()
binding = {a: 1.5, b:2.5}
c = sess.run(add, feed_dict=binding)
print(c)
