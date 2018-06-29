import "../lib/tensorflow"

module tf = tensorflow f32

let layer1 = tf.layers.fully_connected_2d [784, 256]
let layer2 = tf.layers.fully_connected_2d [256, 256]
let output = tf.layers.fully_connected_2d [256, 10]

let nn0   = tf.nn.empty_network()
let nn1   = tf.nn.connect_layer nn0 layer1
let nn2   = tf.nn.connect_layer nn1 layer2
let nn3   = tf.nn.connect_layer nn2 output

let model = tf.nn.init_network_w_rand_norm nn3 1

let main (input: [][]tf.t) (labels:[][]tf.t) =  (tf.optimizer.backprop_batch model input labels)[0]