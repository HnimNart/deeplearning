import "../lib/deep_learning"
module dl = deep_learning f32

let seed = 1

let l1 = dl.layers.Dense (784, 256) dl.nn.identity seed
let l2 = dl.layers.Dense (256, 128) dl.nn.identity seed
let l3 = dl.layers.Dense (128, 10) dl.nn.identity seed

let nn1 = dl.nn.connect_layers l1 l2
let nn  = dl.nn.connect_layers nn1 l3

let main [m][n][d] (input: [m][d]dl.t) (labels: [m][n]dl.t) =
  let batch_size = 100
  let alpha = 0.01
  let nn1 = dl.train.GradientDescent nn alpha input labels batch_size dl.loss.softmax_cross_entropy_with_logits
   in dl.nn.accuracy nn1 (input) (labels) dl.nn.softmax dl.nn.argmax
