-- Small example of training a MLP
--
-- Run 'make' to generate data sets.
-- ==
-- compiled input @ batch_16_mnist_100000_f32.bindata
-- compiled input @ batch_32_mnist_100000_f32.bindata
-- compiled input @ batch_64_mnist_100000_f32.bindata
-- compiled input @ batch_128_mnist_100000_f32.bindata

import "../lib/github.com/HnimNart/deeplearning/deep_learning"
module dl = deep_learning f32

let seed = 1i32

let l1 = dl.layers.dense 784 256 (dl.nn.identity 256) seed
let l2 = dl.layers.dense 256 256 (dl.nn.identity 256) seed
let l3 = dl.layers.dense 256 10 (dl.nn.identity 10) seed

let nn0 = dl.nn.connect_layers l1 l2
let nn  = dl.nn.connect_layers nn0 l3

let main [K] (batch_size: i32) (input:[K][784]dl.t) (labels: [K][10]dl.t) =
  let train = 64000
  let validation = 10000
  let alpha = 0.1
  let nn' = dl.train.gradient_descent nn alpha
            input[:train] labels[:train]
            (i64.i32 batch_size) (dl.loss.softmax_cross_entropy_with_logits 10)
  in dl.nn.accuracy
     nn'
     input[train:train+validation] labels[train:train+validation]
     (dl.nn.softmax 10) dl.nn.argmax
