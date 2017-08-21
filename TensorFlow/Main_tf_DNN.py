import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# number 1 to 10 data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


def tf_ModelAdd_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    W_with_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = W_with_b
    else:
        outputs = activation_function(W_with_b,)
    return outputs

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])


# add output layer
h1 = tf_ModelAdd_layer(xs, 784, 100,  activation_function=tf.nn.sigmoid)
h2 = tf_ModelAdd_layer(h1, 100, 100,  activation_function=tf.nn.sigmoid)
prediction = tf_ModelAdd_layer(h2, 100, 10,  activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss

# optimizer                                             
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#train_step =tf.train.AdamOptimizer(learning_rate=0.1,beta1=0.9,beta2=0.99,epsilon=1e-8,name="Adam").minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

costvalue=[]
iternumber=[]
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(1000)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        a = compute_accuracy(mnist.test.images, mnist.test.labels)
        costvalue.append(a)
        iternumber.append(i)
        print(a)
        
plt.plot(iternumber,costvalue)
plt.scatter(iternumber,costvalue)
plt.ylabel('accuracy')
plt.xlabel('Iternation Time')
plt.show()
