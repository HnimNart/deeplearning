import "../lib/tensorflow"

module tf = tensorflow f64

let layer1 = tf.layers.fully_connected_2d [784, 256]
let layer2 = tf.layers.fully_connected_2d [256, 256]
let output = tf.layers.fully_connected_2d [256, 10]

let nn0   = tf.nn.empty_network()
let nn1   = tf.nn.connect_layer nn0 layer1
let nn2   = tf.nn.connect_layer nn1 layer2
let nn3   = tf.nn.connect_layer nn2 output

let model = tf.nn.init_network_w_rand_norm nn3 1

let main [m][n][d] (input: [m][d]tf.t) (labels: [m][n]tf.t) =
  let data_sets = 500
  let batch_size = 2
  let i = 0
  let tmp = map (\ _ -> 0.0) (0..<length (tf.nn.get_weights model))
  let (nn, _, tmp) = loop (model, i, tmp) while i < data_sets do
           let (train_model, tmp) : (tf.nn.NN, []tf.t) = tf.optimizer.train_batch model input[i:i+batch_size] labels[i:i+batch_size] 0.1
           in (train_model, i + batch_size, tmp)
  in ( tmp[:2000], (tf.nn.get_weights nn)[:10], tf.nn.accuracy nn input labels)
