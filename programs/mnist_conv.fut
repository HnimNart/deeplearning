import "../lib/deep_learning"
module dl = deep_learning f32
let seed = 2

let conv1     = dl.layers.Conv2d (32, 5, 1, 1) dl.nn.relu seed
let max_pool1 = dl.layers.Max_pooling2d (2,2)
let conv2     = dl.layers.Conv2d (64, 3, 1, 32) dl.nn.relu seed
let max_pool2 = dl.layers.Max_pooling2d (2,2)
let flat      = dl.layers.Flatten
let fc        = dl.layers.Dense (1600, 1024) dl.nn.identity seed
let output    = dl.layers.Dense (1024, 10)   dl.nn.identity seed

let nn0   = dl.nn.connect_layers conv1 max_pool1
let nn1   = dl.nn.connect_layers nn0 conv2
let nn2   = dl.nn.connect_layers nn1 max_pool2
let nn3   = dl.nn.connect_layers nn2 flat
let nn4   = dl.nn.connect_layers nn3 fc
let nn    = dl.nn.connect_layers nn4 output

let main [m][d][n] (input: [m][d]dl.t) (labels: [m][n]dl.t) =
  let input' = map (\img -> [unflatten 28 28 img]) input
  let batch_size = 100
  let alpha = 0.01
  let nn' = dl.train.GradientDescent nn alpha input' labels batch_size dl.loss.softmax_cross_entropy_with_logits
  in nn'.4
  -- let j = 0
  -- let size = 1000
  -- let acc = 0
  -- let (acc, _) = loop (acc, j) while j < length input do
  --                let acc = acc + dl.nn.accuracy nn' (input'[j:j+size]) (labels[j:j+size]) (dl.nn.softmax) (dl.nn.argmax)
  --                in (acc, j + size)
  -- in (acc)
