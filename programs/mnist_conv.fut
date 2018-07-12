import "../lib/tensorflow"

module tf = tensorflow f32
let seed = 1

let conv1     = tf.layers.Conv2d (32, 5, 1, 1) tf.nn.relu seed
let max_pool1 = tf.layers.Max_pooling2d (2,2)
let conv2     = tf.layers.Conv2d (64, 3, 1, 32) tf.nn.relu seed
let max_pool2 = tf.layers.Max_pooling2d (2,2)
let flat      = tf.layers.Flatten
let fc        = tf.layers.Dense (1600, 1024) tf.nn.identity seed
let output    = tf.layers.Dense (1024, 10)   tf.nn.identity seed

let nn0   = tf.nn.connect_layers conv1 max_pool1
let nn1   = tf.nn.connect_layers nn0 conv2
let nn2   = tf.nn.connect_layers nn1 max_pool2
let nn3   = tf.nn.connect_layers nn2 flat
let nn4   = tf.nn.connect_layers nn3 fc
let nn    = tf.nn.connect_layers nn4 output


let main [m][d][n] (input: [m][d]tf.t) (labels: [m][n]tf.t) =
  let input' = map (\img -> [unflatten 28 28 img]) input
  let n = 64000
  let batch_size = 100
  let alpha = 0.01
  let nn' = tf.train.GradientDescent nn alpha input' labels batch_size tf.loss.softmax_cross_entropy_with_logits in nn'.4
  -- in tf.nn.accuracy nn' (input'[:2000]) (labels[:2000]) (tf.nn.softmax) (tf.nn.argmax)