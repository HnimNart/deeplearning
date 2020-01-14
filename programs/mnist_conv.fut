-- Small example of training a convolutional network
--
-- Run 'make' to generate data sets.
-- ==
-- compiled input @ batch_16_mnist_100000_f32.bindata
-- compiled input @ batch_32_mnist_100000_f32.bindata
-- compiled input @ batch_64_mnist_100000_f32.bindata
-- compiled input @ batch_128_mnist_100000_f32.bindata

import "../lib/github.com/HnimNart/deeplearning/deep_learning"
module dl = deep_learning f32

let seed = 1

let identity (d: i32) : activation_func ([d]dl.t) =
  dl.nn.identity : activation_func ([d]dl.t)

let relu (d: i32) : activation_func ([d]dl.t) =
  dl.nn.relu : activation_func ([d]dl.t)

let (>>) = dl.nn.connect_layers

let nn = dl.layers.conv2d 1 28 28 5 1 32 24 24 relu seed
         >> dl.layers.max_pooling2d 32 24 24 12 12
         >> dl.layers.conv2d 32 12 12 3 1 64 10 10 relu seed
         >> dl.layers.max_pooling2d 64 10 10 5 5
         >> dl.layers.flatten 64 5 5 1600
         >> dl.layers.dense 1600 1024 dl.nn.identity seed
         >> dl.layers.dense 1024 10 dl.nn.identity seed

let main [m] (batch_size: i32) (input: [m][]dl.t) (labels: [m][]dl.t) =
  let input' = map (\img -> [unflatten 28 28 img]) input
  let train = 64000
  let validation = 10000
  let alpha = 0.1
  let nn' = dl.train.gradient_descent nn alpha
            input'[:train] labels[:train]
            batch_size dl.loss.softmax_cross_entropy_with_logits
  in dl.nn.accuracy
     nn'
     input'[train:train+validation] labels[train:train+validation]
     dl.nn.softmax dl.nn.argmax
