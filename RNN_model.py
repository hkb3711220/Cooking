import tensorflow as tf
from RNN_model_preprocess import test_data, train_data, train_label, train_data_leg, max_data_leg, ingredient_dict,cuisine_dict
from tensorflow.nn.rnn_cell import GRUCell, DropoutWrapper, MultiRNNCell
import numpy as np

def embedding_variable(vol_size, embed_size):
    init_embed = tf.random_uniform([vol_size, embed_size])
    return tf.Variable(init_embed)

def next_batch(num, data, labels, sequence_length):

    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    seq_leg_shuffle = [sequence_length[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle), np.asarray(seq_leg_shuffle)

num_layers = 3
batch_size = 10
num_class = len(cuisine_dict)
embed_ingred_size = len(ingredient_dict)
embed_size = 100

#If time_major == False (default), this must be a Tensor of shape: [batch_size, max_time, ...],
#or a nested tuple of such elements.
#If time_major == True, this must be a Tensor of shape: [max_time, batch_size, ...],
#or a nested tuple of such elements.

x = tf.placeholder(dtype=tf.int32, shape=[None, None])
y = tf.placeholder(dtype=tf.int64, shape=[None])
sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])
keep_prob = tf.placeholder(dtype=tf.float32)
num_units = 100
n_epoch = 10

with tf.variable_scope('embedding'):
    rnn_input = tf.contrib.layers.embed_sequence(x,
                                                 vocab_size=embed_ingred_size,
                                                 embed_dim=embed_size)

with tf.variable_scope('rnn'):
    cell = GRUCell(num_units)
    drop_cell = DropoutWrapper(cell, output_keep_prob=keep_prob)
    stack_cell = MultiRNNCell([drop_cell for _ in range(num_layers)])

    outputs, states = tf.nn.dynamic_rnn(stack_cell,
                                       rnn_input,
                                       dtype=tf.float32,
                                       sequence_length=sequence_length)
    # ★Attention
    # 'outputs' is a tensor of shape [batch_size, max_time, num_of_units]
    # 'state' is a N-tuple where N is the number of GRUCells containing a
    # tf.contrib.rnn.GRUcells for each cell

with tf.variable_scope('full_connected'):
    state = states[-1]
    fc = tf.contrib.layers.fully_connected(state, num_class, activation_fn=None)

with tf.name_scope('train'):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fc)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(0.001)
    training_op = optimizer.minimize(loss)

with tf.name_scope('accuarcy'):
    predicted = tf.nn.softmax(fc)
    correct_pred = tf.equal(tf.argmax(predicted, 1), y)
    accuarcy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
init.run()

for epoch in range(n_epoch):
    X_batch, y_batch, seq_leg_batch = next_batch(batch_size, train_data, train_label, train_data_leg)
    print(y_batch)
    #output = sess.run([xentropy], feed_dict={x:X_batch,
                                     #y:y_batch,
                                     #sequence_length:seq_leg_batch,
                                     #keep_prob:0.5})
    #accuarcy_train = sess.run(accuarcy, feed_dict={X:X_batch,
                                                   #y:y_batch,
                                                   #sequence_length:seq_leg_batch,
                                                   #keep_prob:1.0})
    #print(output)
