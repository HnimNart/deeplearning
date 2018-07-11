import "../lib/tensorflow"

module tf = tensorflow f32
let seed = 1

let conv1     = tf.layers.Conv2d (32, 5, 1, 1) tf.activation.Relu_1d seed
let max_pool1 = tf.layers.Max_pooling2d (2,2)
let conv2     = tf.layers.Conv2d (64, 3, 1, 32) tf.activation.Relu_1d seed
let max_pool2 = tf.layers.Max_pooling2d (2,2)
let flat      = tf.layers.Flatten
let fc        = tf.layers.Dense (1600, 1024) tf.activation.Identity_1d seed
let output    = tf.layers.Dense (1024, 10)   tf.activation.Identity_1d seed

let nn0   = tf.nn.connect_layers conv1 max_pool1
let nn1   = tf.nn.connect_layers nn0 conv2
let nn2   = tf.nn.connect_layers nn1 max_pool2
let nn3   = tf.nn.connect_layers nn2 flat
let nn4   = tf.nn.connect_layers nn3 fc
let nn    = tf.nn.connect_layers nn4 output

let get_dims (X:[][][][]f32) =
  (length X, length X[0], length X[0,0], length X[0,0,0])



let main [m][d][n] (input: [m][d]tf.t) (labels: [m][n]tf.t) =
  let n = 64000
  let batch_size = 128
  let alpha = 0.01
  let input' = map (\img -> [unflatten 28 28 img]) input[:n]
  let nn' = tf.train.GradientDescent nn alpha input' labels[:n] batch_size tf.loss.Softmax_cross_entropy_with_logits_2d.2
  in nn'.4


  -- let test_input = map (\img -> [unflatten 28 28 img]) input-- [n:n+1000]
  -- let test_label = labels--[n:n+1000]
  -- in tf.nn.accuracy nn' (test_input) (test_label) (tf.activation.Softmax_2d.1) (tf.nn.argmax)
