import "../lib/network"
import "../lib/activations"
import "../lib/classifications"

module tf = network f32
module act = activations f32
module cross = softmax_cross_entropy_with_logits f32
module softmax = softmax_stable f32

let l1    = tf.dense.layer (784, 256) act.Identity_1d false
let l2    = tf.dense.layer (256, 256) act.Identity_1d false
let l3    = tf.dense.layer (256, 10) act.Identity_1d true
let nn'   = tf.combine l1 l2
let model = tf.combine nn' l3


let main [m][n][d] (input: [m][d]tf.t) (labels: [m][n]tf.t) =
  let nn = tf.train model (input) (labels) 0.01
  in (tf.accuracy nn (input) (labels))
      -- tf.accuracy model (transpose input) (transpose labels))
