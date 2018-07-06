import "../lib/network"
import "../lib/activations"
import "../lib/classifications"
import "../lib/optimizers"

module tf = network f32
module act = activations f32
module sgd = sgd f32

let l1    = tf.dense.layer (784, 256) act.Identity_1d false
let l2    = tf.dense.layer (256, 256) act.Identity_1d false
let l3    = tf.dense.layer (256, 10) act.Identity_1d true
let nn'   = tf.combine l1 l2
let model = tf.combine nn' l3




let main [m][n][d] (input: [m][d]tf.t) (labels: [m][n]tf.t) =
  let nn = sgd.train model (input) (labels) in nn.4
  -- in (tf.accuracy nn (input) (labels))
      -- tf.accuracy model (transpose input) (transpose labels))



--- Todo
-- 1. remove last layer indicator
-- 2. implement optimizer functions
-- 3. implement loss function
-- 4. conv2d