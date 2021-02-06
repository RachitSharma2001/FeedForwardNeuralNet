import tensorflow as tf
import numpy as np
from random import *

# ------------------ Initialization ------------------

# Define the training and testing data
inputs = [[5.5,2.6,4.4,1.2],[4.6,3.2,1.4,0.2],[6.3,2.9,5.6,1.8],[5,3,1.6,0.2],[6.8,3,5.5,2.1],[6.3,2.5,4.9,1.5],[5.8,2.7,4.1,1],[5.2,3.5,1.5,0.2],[5.8,2.6,4,1.2],[5.1,3.8,1.9,0.4],[7.2,3.6,6.1,2.5],[5.2,3.4,1.4,0.2],[4.4,3,1.3,0.2],[5.7,3.8,1.7,0.3],[4.8,3,1.4,0.1],[5.5,3.5,1.3,0.2],[6.5,2.8,4.6,1.5],[5,2,3.5,1],[4.9,3,1.4,0.2],[6,2.2,4,1],[7.1,3,5.9,2.1],[6.7,3.3,5.7,2.1],[6.7,3.1,4.4,1.4],[6.3,3.4,5.6,2.4],[5,3.2,1.2,0.2],[5.6,3,4.5,1.5],[4.9,3.1,1.5,0.1],[6.1,2.8,4.7,1.2],[5.6,2.5,3.9,1.1],[5,3.6,1.4,0.2],[6.7,3.3,5.7,2.5],[6.4,3.2,5.3,2.3],[6.6,3,4.4,1.4],[6.9,3.2,5.7,2.3],[5.6,3,4.1,1.3],[6.4,2.7,5.3,1.9],[4.7,3.2,1.6,0.2],[5.4,3.9,1.7,0.4],[6.7,3,5.2,2.3],[6.1,3,4.9,1.8],[5.4,3.9,1.3,0.4],[4.8,3.4,1.6,0.2],[6.1,2.9,4.7,1.4],[5.4,3,4.5,1.5],[5,3.3,1.4,0.2],[6,3,4.8,1.8],[4.3,3,1.1,0.1],[6.9,3.1,5.4,2.1],[6.3,2.8,5.1,1.5],[5.5,2.4,3.8,1.1],[6.1,3,4.6,1.4],[5.5,2.4,3.7,1],[5.8,2.7,5.1,1.9],[7.7,2.6,6.9,2.3],[5.6,2.8,4.9,2],[5.1,3.8,1.6,0.2],[5.1,3.5,1.4,0.2],[5.7,4.4,1.5,0.4],[6.1,2.8,4,1.3],[5.7,2.8,4.1,1.3],[4.9,2.5,4.5,1.7],[6.3,2.7,4.9,1.8],[5,3.5,1.6,0.6],[5.9,3,4.2,1.5],[5.6,2.9,3.6,1.3],[7.2,3,5.8,1.6],[6.3,3.3,6,2.5],[5,3.5,1.3,0.3],[6.2,2.2,4.5,1.5],[5.1,3.8,1.5,0.3],[5.9,3.2,4.8,1.8],[6.8,2.8,4.8,1.4],[6.9,3.1,4.9,1.5],[6.5,3,5.5,1.8],[5.3,3.7,1.5,0.2],[5.8,4,1.2,0.2],[6.2,2.8,4.8,1.8],[7.7,3,6.1,2.3],[6.5,3,5.2,2],[5.7,2.9,4.2,1.3],[5.5,2.5,4,1.3],[5.1,2.5,3,1.1],[7.9,3.8,6.4,2],[4.4,2.9,1.4,0.2],[6.2,2.9,4.3,1.3],[5.1,3.7,1.5,0.4],[5.1,3.4,1.5,0.2],[5,2.3,3.3,1],[6.7,3.1,4.7,1.5],[6.4,3.2,4.5,1.5],[5.1,3.5,1.4,0.3],[6,2.2,5,1.5],[5.5,2.3,4,1.3],[6.9,3.1,5.1,2.3],[5.7,2.6,3.5,1],[6.2,3.4,5.4,2.3],[5.4,3.7,1.5,0.2],[6.7,3,5,1.7],[7.4,2.8,6.1,1.9],[6.8,3.2,5.9,2.3],[4.8,3,1.4,0.3],[4.6,3.6,1,0.2],[5.4,3.4,1.7,0.2],[6,3.4,4.5,1.6],[5.1,3.3,1.7,0.5],[5.2,4.1,1.5,0.1],[6.6,2.9,4.6,1.3],[6.3,2.3,4.4,1.3],[5.5,4.2,1.4,0.2],[5.9,3,5.1,1.8],[6.5,3.2,5.1,2],[6.3,2.5,5,1.9],[5,3.4,1.5,0.2],[4.6,3.1,1.5,0.2],[4.9,2.4,3.3,1],[7,3.2,4.7,1.4],[7.6,3,6.6,2.1],[5.4,3.4,1.5,0.4],[5.8,2.7,5.1,1.9],[6.4,3.1,5.5,1.8],[7.2,3.2,6,1.8],[4.8,3.4,1.9,0.2],[7.7,3.8,6.7,2.2],[5,3.4,1.6,0.4],[6.7,3.1,5.6,2.4],[4.5,2.3,1.3,0.3],[6.5,3,5.8,2.2],[4.8,3.1,1.6,0.2],[5.2,2.7,3.9,1.4],[6.4,2.8,5.6,2.1],[5.7,2.8,4.5,1.3],[6.4,2.8,5.6,2.2],[6.7,2.5,5.8,1.8],[6,2.9,4.5,1.5],[6,2.7,5.1,1.6],[7.7,2.8,6.7,2],[4.7,3.2,1.3,0.2],[5.7,3,4.2,1.2],[5.8,2.7,3.9,1.2],[5.8,2.8,5.1,2.4],[6.1,2.6,5.6,1.4],[5.7,2.5,5,2],[6.4,2.9,4.3,1.3],[7.3,2.9,6.3,1.8],[6.3,3.3,4.7,1.6],[4.6,3.4,1.4,0.3],[4.4,3.2,1.3,0.2],[5.6,2.7,4.2,1.3],[4.9,3.1,1.5,0.1],[4.9,3.1,1.5,0.1]]
outputs = [[0,0,1],[0,1,0],[1,0,0],[0,1,0],[1,0,0],[0,0,1],[0,0,1],[0,1,0],[0,0,1],[0,1,0],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1],[0,1,0],[0,0,1],[1,0,0],[1,0,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1],[0,1,0],[0,0,1],[0,0,1],[0,1,0],[1,0,0],[1,0,0],[0,0,1],[1,0,0],[0,0,1],[1,0,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1],[0,1,0],[1,0,0],[0,1,0],[1,0,0],[1,0,0],[0,0,1],[0,0,1],[0,0,1],[1,0,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1],[1,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,1],[1,0,0],[1,0,0],[0,1,0],[0,0,1],[0,1,0],[0,0,1],[0,0,1],[0,0,1],[1,0,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0],[1,0,0],[0,0,1],[0,0,1],[0,0,1],[1,0,0],[0,1,0],[0,0,1],[0,1,0],[0,1,0],[0,0,1],[0,0,1],[0,0,1],[0,1,0],[1,0,0],[0,0,1],[1,0,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1],[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,1,0],[0,1,0],[0,0,1],[0,0,1],[0,1,0],[1,0,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1],[1,0,0],[0,1,0],[1,0,0],[1,0,0],[1,0,0],[0,1,0],[1,0,0],[0,1,0],[1,0,0],[0,1,0],[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,0,1],[1,0,0],[1,0,0],[0,0,1],[0,0,1],[1,0,0],[0,1,0],[0,0,1],[0,0,1],[1,0,0],[1,0,0],[1,0,0],[0,0,1],[1,0,0],[0,0,1],[0,1,0],[0,1,0],[0,0,1],[0,1,0],[0,1,0]]

# Defining important variables
learning_rate = 0.5
epochs = 300
batch_size = 100
train_size = 80
test_size = 70
input_layer_size = 4
hidden_layer_size = 30
output_layer_size = 3

# Input data
x = tf.placeholder(tf.float64, [None, input_layer_size])
# Expected output
y = tf.placeholder(tf.float64, [None, output_layer_size])

# Declare the weights and biases connecting the
# input layer with the hidden layer
w1 = tf.Variable(np.array([[uniform(-0.5, 0.5) for _ in range(hidden_layer_size)] for _ in range(input_layer_size)]), name='w1')
b1 = tf.Variable(np.array([uniform(-0.5, 0.5) for _ in range(hidden_layer_size)]), name='b1')

# Declare the weights and biases connecting the
# hidden layer with the output layer
w2 = tf.Variable(np.array([[uniform(-0.5, 0.5) for _ in range(output_layer_size)] for _ in range(hidden_layer_size)]), name='w2')
b2 = tf.Variable(np.array([uniform(-0.5, 0.5) for _ in range(output_layer_size)]), name='b2')

# ------------------ Operations ------------------

# Get the output for the hidden layer
hidden_output = tf.add(tf.matmul(x, w1), b1)
hidden_output = tf.sigmoid(hidden_output)

# Get the output for the output layer
y_ = tf.add(tf.matmul(hidden_output, w2), b2)
y_ = tf.sigmoid(y_)

# Calculate the error
difference = tf.subtract(y, y_)
# Mean Squared Error -> (1/n) * sum of differences
error = tf.reduce_mean(tf.reduce_sum(tf.multiply(difference, difference), axis=1))

# add the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(error)

# set up init operator
init_op = tf.global_variables_initializer()

# store a vector of True and False values
# for each index it has either True(if y[index]'s largest value occurs
# at the same spot as y_[index]'s largest value)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# Now we need to define a way to measure the accuracy of our algorithm
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

# ------------------ Run the session ------------------

with tf.Session() as sess:
    # First initialize everything
    sess.run(init_op)

    # Now test and train
    for epoch in range(epochs):
        # Testing portion
        for data in range(train_size):
            batch_input = [[inputs[data][i] for i in range(input_layer_size)] for _ in range(1)]
            batch_output = [[outputs[data][i] for i in range(output_layer_size)] for _ in range(1)]
            _ = sess.run([optimizer], feed_dict={x: batch_input, y: batch_output})
        # Training portion
        test_input = [[inputs[i+train_size][j] for j in range(input_layer_size)] for i in range(test_size)]
        test_output = [[outputs[i+train_size][j] for j in range(output_layer_size)] for i in range(test_size)]
        print 'Epoch #: {} has accuracy {}'.format(str(epoch), sess.run(accuracy, feed_dict={x: test_input, y: test_output}))
        
        
        
