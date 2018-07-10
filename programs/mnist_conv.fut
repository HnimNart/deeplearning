import "../lib/tensorflow"

module tf = tensorflow f32
let seed = 1

let conv1     = tf.layers.Conv2d (32, 5, 1) tf.activation.Relu_1d  seed
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
  let i = 00
  let n = 1000
  let batch_size = 100
  let tmp = map (\img -> [unflatten 28 28 img]) input
  let (nnf,nnb, nnu,w) = nn

  let j = 1
  let (w', _) = loop (w, j) while j < 100 do
    let (w', _) = loop (w, i) while i < n do
      let input' = tmp[i:i+batch_size]
      let label' = labels[i:i+batch_size]
      let (os,out) = nnf w input'
      let error = tf.loss.Softmax_cross_entropy_with_logits_2d.2 out label'
      let (_, g) = nnb w os (transpose error)
      let w' = nnu (0.0001) w g
    in (w', i + batch_size)
  in (w' , j + 1)

  in (tf.nn.accuracy (nnf, nnb, nnu , w') tmp[:n] labels[:n] tf.activation.Softmax_2d.1)
