import "../lib/github.com/HnimNart/deeplearning/deep_learning"

--- Small example of training a convolutional network
--- Data can be downloaded at
--- http://napoleon.hiperfit.dk/~HnimNart/mnist_data/mnist_100000_f32.bindata
--- Containing 100000 pairs of images and labels

module dl = deep_learning f32
let seed = 1

let conv1     = dl.layers.conv2d (32, 5, 1, 1) dl.nn.relu seed
let max_pool1 = dl.layers.max_pooling2d (2,2)
let conv2     = dl.layers.conv2d (64, 3, 1, 32) dl.nn.relu seed
let max_pool2 = dl.layers.max_pooling2d (2,2)
let flat      = dl.layers.flatten
let fc        = dl.layers.dense (1600, 1024) dl.nn.identity seed
let output    = dl.layers.dense (1024, 10)   dl.nn.identity seed

let nn0   = dl.nn.connect_layers conv1 max_pool1
let nn1   = dl.nn.connect_layers nn0 conv2
let nn2   = dl.nn.connect_layers nn1 max_pool2
let nn3   = dl.nn.connect_layers nn2 flat
let nn4   = dl.nn.connect_layers nn3 fc
let nn    = dl.nn.connect_layers nn4 output

let main [m] (batch_size: i32) (input: [m][]dl.t) (labels: [m][]dl.t) =
  let input' = map (\img -> [unflatten 28 28 img]) input
  let train = 64000
  let validation = 10000
  let alpha = 0.1
  let nn' = dl.train.gradient_descent nn alpha
            input'[:train] labels[:train]
            batch_size dl.loss.softmax_cross_entropy_with_logits
  in dl.nn.accuracy nn'
     input'[train:train+validation]
     labels[train:train+validation]
     (dl.nn.softmax) (dl.nn.argmax)

-- ==
-- compiled input @ batch_16_mnist_100000_f32.bindata
-- compiled input @ batch_32_mnist_100000_f32.bindata
-- compiled input @ batch_64_mnist_100000_f32.bindata
-- compiled input @ batch_128_mnist_100000_f32.bindata
