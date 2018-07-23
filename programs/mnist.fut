import "../lib/deep_learning"
module dl = deep_learning f32

let seed = 1

let l1 = dl.layers.dense (784, 256) dl.nn.identity seed
let l2 = dl.layers.dense (256, 256) dl.nn.identity seed
let l3 = dl.layers.dense (256, 10) dl.nn.identity seed

let nn1 = dl.nn.connect_layers l1 l2
let nn  = dl.nn.connect_layers nn1 l3

let main [m][n][d] (input: [m][d]dl.t) (labels: [m][n]dl.t) =
  let n = 32000
  let batch_size = 64
  let alpha = 0.1
  let nn1 = dl.train.GradientDescent nn alpha input[:n] labels[:n] batch_size dl.loss.softmax_cross_entropy_with_logits
  in nn1.4
   -- in dl.nn.accuracy nn1 (input) (labels) dl.nn.softmax dl.nn.argmax
