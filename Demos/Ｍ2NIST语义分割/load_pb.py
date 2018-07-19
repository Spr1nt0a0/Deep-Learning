import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('train.csv').as_matrix()
from tensorflow.core.framework import graph_pb2
graph_def = graph_pb2.GraphDef()

def load_graph():
    with open('./checkpoints/frozen_graph.pb', "rb") as f:
        graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
            return graph

frozen_graph = load_graph()

with tf.Session(graph=frozen_graph) as sess:
    # Get placeholder and output tensors.
    # Tensor names are derived from the operation that produced them
    # We named operations while building the graph NOT tensors.
    input_ph = tf.get_default_graph().get_tensor_by_name('input_image:0')
    # labels_ph = tf.get_default_graph().get_tensor_by_name('labels:0')
    # learning_rate_ph = tf.get_default_graph().get_tensor_by_name('learning_rate:0')
    output_prediction = tf.get_default_graph().get_tensor_by_name('predictions:0')

    # Select 5 random images
    indices = np.random.randint(0, len(data), [5])
    X = (data[indices, 1:] - 127) / 127
    y = data[indices, 0].reshape([-1, 1])

    predictions = sess.run(output_prediction, feed_dict={input_ph: X})
    y_predicted = np.argmax(predictions, axis=1).flatten()
    print(y_predicted)

for i in range(len(X)):
    plt.figure(figsize=(0.25,0.25))
    plt.imshow(X[i].reshape([28,28]),cmap='gray')
    plt.title('Predicted as {}'.format(y_predicted[i]))
    plt.show()