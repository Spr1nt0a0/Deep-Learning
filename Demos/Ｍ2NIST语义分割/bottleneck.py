import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

X = np.load('combined.npy')

X = (X.astype(np.float32)-127)/127.0
from tensorflow.core.framework import graph_pb2
graph_def = graph_pb2.GraphDef()

def load_graph(path_protobuf):
    with open(path_protobuf, "rb") as f:
        graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            # Createa new placeholder
            input_big = tf.placeholder(dtype=tf.float32, shape=(None, 64,84,1), name='input_image_big')
            # Import the graph and replace the reshape node with new placeholder.
            tf.import_graph_def(graph_def, name="", input_map={"reshaped_image": input_big})
            return graph

frozen_graph = load_graph('./checkpoints/frozen_graph.pb')

## Uncomment to view all names of all nodes.
for op in frozen_graph.get_operations():
   print(op.name)

def get_next_batch(X,batch_sz):
    for start_offset in range(0, len(X), batch_sz):
        end_offfset = min(start_offset+batch_sz, len(X))
        yield X[start_offset:end_offfset]


bottleneck_features = []
with tf.Session(graph=frozen_graph) as sess:
    # Get placeholder and output tensors of last pooling layer.
    # Tensor names are derived from the operation that produced them
    # We named layers while building the graph, NOT tensors.
    input_ph = tf.get_default_graph().get_tensor_by_name('input_image_big:0')
    bottleneck_tensor = tf.get_default_graph().get_tensor_by_name('poool2/MaxPool:0')

    # Loop over the whole dataset
    for X_batch in get_next_batch(X, 128):
        # Add empty dimension to match the dimesnion of the placeholder.
        X_batch = np.expand_dims(X_batch, 3)

        bottleneck_batch = sess.run(bottleneck_tensor, feed_dict={input_ph: X_batch})
        bottleneck_features.extend(bottleneck_batch)

np.save('bottleneck.npy',bottleneck_features)