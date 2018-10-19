import tensorflow as tf
from RNN_model_preprocess import test_data, test_data_leg, cuisine_dict, test_data_ids
import pandas as pd

model = tf.train.import_meta_graph('./my_model.ckpt.meta')

graph = tf.get_default_graph()
x = graph.get_tensor_by_name('Placeholder:0')
sequence_length = graph.get_tensor_by_name('Placeholder_2:0')
keep_prob = graph.get_tensor_by_name('Placeholder_3:0')
output = graph.get_tensor_by_name('full_connected/fully_connected/BiasAdd:0')
result = tf.argmax((tf.nn.softmax(output)), 1)

with tf.Session() as sess:
    #for op in tf.get_default_graph().get_operations():
        #print(op.name)

    model.restore(sess, './my_model.ckpt')
    predictions = sess.run([result], feed_dict={x:test_data,
                                               sequence_length:test_data_leg,
                                               keep_prob:1.0})

def get_keys_from_value(d, val):
    result = [k for k, v in d.items() if v == val]
    return result[0]

submission = []
for elements in predictions:
    for element in elements:
        submission.append(get_keys_from_value(cuisine_dict, element))

submission = pd.DataFrame({'id': test_data_ids,
                           'cuisine': submission})

submission.to_csv('output.csv', index=False)
