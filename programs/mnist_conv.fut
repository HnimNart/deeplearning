import "../lib/tensorflow"

module tf = tensorflow f32
let seed = 1

let conv1     = tf.layers.Conv2d (32, 5, 1) tf.activation.Relu_1d seed
let max_pool1 = tf.layers.Max_pooling2d (2,2)
let conv2     = tf.layers.Conv2d (64, 3, 1) tf.activation.Relu_1d seed
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


let main [m][d][n] (input: [m][d]tf.t) (labels: [m][n]tf.t) =
  let n = 1000
  let batch_size = 10
  let alpha = 0.0001
  let alpha1 = 0.0001
  let alpha2 = 0.0001
  let alpha3 = 0.00005
  let alpha4 = 0.00005
  let input' = map (\img -> [unflatten 28 28 img]) input[:n]
  let nn = tf.train.GradientDescent nn alpha input' labels[:n] batch_size tf.loss.Softmax_cross_entropy_with_logits_2d.2
  let nn = tf.train.GradientDescent nn alpha1 input' labels[:n] batch_size tf.loss.Softmax_cross_entropy_with_logits_2d.2
  let nn = tf.train.GradientDescent nn alpha2 input' labels[:n] batch_size tf.loss.Softmax_cross_entropy_with_logits_2d.2
  let nn = tf.train.GradientDescent nn alpha2 input' labels[:n] batch_size tf.loss.Softmax_cross_entropy_with_logits_2d.2
  let nn = tf.train.GradientDescent nn alpha3 input' labels[:n] batch_size tf.loss.Softmax_cross_entropy_with_logits_2d.2
  let nn = tf.train.GradientDescent nn alpha4 input' labels[:n] batch_size tf.loss.Softmax_cross_entropy_with_logits_2d.2
  let nn = tf.train.GradientDescent nn alpha4 input' labels[:n] batch_size tf.loss.Softmax_cross_entropy_with_logits_2d.2
  in tf.nn.accuracy nn (input') (labels[:n]) (tf.activation.Softmax_2d.1) (tf.nn.argmax)
