import "../lib/github.com/HnimNart/deeplearning/deep_learning"
module dl = deep_learning f32

--- Small example of training a MLP
--- Data can be downloaded at
--- http://napoleon.hiperfit.dk/~HnimNart/mnist_data/mnist_100000_f32.bindata
--- Containing 100000 pairs of images and labels

let seed = 1

let l1 = dl.layers.dense (784, 256) dl.nn.identity seed
let l2 = dl.layers.dense (256, 256) dl.nn.identity seed
let l3 = dl.layers.dense (256, 10) dl.nn.identity seed

let nn0 = dl.nn.connect_layers l1 l2
let nn  = dl.nn.connect_layers nn0 l3

let main [m] (input:[m][]dl.t) (labels:[m][]dl.t) =
  let train = 64000
  let validation = 10000
  let batch_size = 128
  let alpha = 0.1
  let nn' = dl.train.gradient_descent nn alpha
            input[:train] labels[:train]
            batch_size dl.loss.softmax_cross_entropy_with_logits
  in dl.nn.accuracy nn' input[train:train+validation]
     labels[train:train+validation] dl.nn.softmax dl.nn.argmax
