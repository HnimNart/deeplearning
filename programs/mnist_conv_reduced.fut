import "../lib/network"
import "../lib/activations"
import "../lib/layers/conv2d"
import "../lib/layers/max_pooling"

module tf     = network f32
module act    = activations f32
module conv2d = conv2d f32
module max    = max_pooling_2d f32

let conv1 = conv2d.layer (64, 3, 1) act.Relu_1d
let max_pooling1 = max.layer (2,2) ((), ())

let nn = tf.combine conv1 max_pooling1

let main [m][d] (input: [m][d]tf.t) =
  let i = 0
  let batch_size = 10
  let (nnf,_,_ ,w) = nn
  let tmp = map (\img -> [unflatten 28 28 img]) input
  let input' = tmp[i:i+batch_size]
  let (_,output) = nnf w input'
  in output
