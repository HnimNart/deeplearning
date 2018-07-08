import "../lib/network"
import "../lib/activations"
import "../lib/classifications"
import "../lib/optimizers"
import "../lib/layers/layers"
import "../lib/layers/conv2d"
import "../lib/layers/max_pooling"
import "../lib/layers/reshape"
import "../lib/loss"
import "/futlib/linalg"

module tf = network f64
module layers = layers f64
module act = activations f64
module sgd = sgd f64
module conv2d = conv2d f64
module max = max_pooling_2d f64
module flat = flatten f64
module loss = loss f64
module class = classification f64
module lalg = linalg f64


let conv1 = conv2d.layer (32, 5, 1) act.Relu_1d
let max_pooling1 = max.layer (2,2) ((), ())
let conv2 = conv2d.layer (64, 3, 1) act.Relu_1d
let max_pooling2 = max.layer (2,2) ((), ())
let flat2 = flat.layer () ((),())
let dense = layers.Dense (1600, 1024) act.Identity_1d
let output= layers.Dense (1024, 10) act.Identity_1d

let nn1 = tf.combine conv1 max_pooling1
let nn2 = tf.combine nn1 conv2
let nn3 = tf.combine nn2 max_pooling2
let nn4 = tf.combine nn3 flat2
let nn5 = tf.combine nn4 dense
let nn = tf.combine nn5 output

let get_dims (X:[][][]f64) =
  (length X, length X[0], length X[0,0])

let main [m][d][n] (input: [m][d]tf.t) (labels: [m][n]tf.t) =
  let i = 0
  let n = 64000
  let (nnf,nnb, nnu,w) = nn
  let (w', _) = loop (w, i) while i < n - 1 do
    let input' = [unflatten 28 28 input[i]]
    let label' = labels[i:i+1]
    let (os,out) = nnf w input'
    let error = loss.softmax_cross_entropy_with_logits label' out
    let (_, g) = nnb true w os error
    let w' = nnu 0.001 w g
    in (w', i + 1)

  let acc: f64 = 0
  let i = i
  let (acc',_) = loop (acc, i) while i < n -1  do
                 let input' = [unflatten 28 28 input[i]]
                 let label' = labels[i:i+1]
                 let acc = acc + tf.accuracy (nnf, nnb, nnu, w') input' label'
                 in (acc, i+ 1)
  in acc'/f64.(i32 n)
