import "../lib/network"
import "../lib/activations"
import "../lib/classifications"
import "../lib/optimizers"
import "../lib/layers"

module tf = network f32
module layers = layers f32
module act = activations f32
module sgd = sgd f32

let l1    = layers.Dense (784, 256) act.Identity_1d
let l2    = layers.Dense (256, 128) act.Identity_1d
let l3    = layers.Dense (128, 10) act.Identity_1d
let nn'   = tf.combine l1 l2
let model = tf.combine nn' l3


let main [m][n][d] (input: [m][d]tf.t) (labels: [m][n]tf.t) =
  let trainer = sgd.sgd model 0.01 input[:64000] labels[:64000]
  let nn = trainer 128
  in tf.accuracy nn (input) (labels)
      -- tf.accuracy model (transpose input) (transpose labels))



--- Todo
-- 1. remove last layer indicator - Done
-- 2. implement optimizer functions -- can't be done bc. of functional types != abstract types
-- 2a. add alpha into - Done
-- 3. implement loss function
-- 4. conv2d