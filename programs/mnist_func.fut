import "../lib/tensorflow"
module tf = tensorflow f32

let seed = 10

let l1 = tf.layers.Dense (784, 256) tf.activation.Identity_1d seed
let l2 = tf.layers.Dense (256, 128) tf.activation.Identity_1d seed
let l3 = tf.layers.Dense (128, 10) tf.activation.Identity_1d seed

let nn1 = tf.nn.connect_layers l1 l2
let nn  = tf.nn.connect_layers nn1 l3


let main [m][n][d] (input: [m][d]tf.t) (labels: [m][n]tf.t) =
  let batch_size = 100
  let alpha = 0.01
  let nn = tf.train.GradientDescent nn alpha input labels batch_size tf.loss.Softmax_cross_entropy_with_logits_2d.2
   in tf.nn.accuracy nn (input) (labels) tf.activation.Softmax_2d.1
