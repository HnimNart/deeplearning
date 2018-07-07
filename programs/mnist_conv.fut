import "../lib/network"
import "../lib/activations"
import "../lib/classifications"
import "../lib/optimizers"
import "../lib/layers/layers"
import "../lib/layers/conv2d"
import "../lib/layers/max_pooling"
import "../lib/layers/reshape"
import "../lib/loss"

module tf = network f32
module layers = layers f32
module act = activations f32
module sgd = sgd f32
module conv2d = conv2d f32
module max = max_pooling_2d f32
module flat = flatten f32
module loss = loss f32
module class = classification f32

let c = conv2d.layer (1, 3) act.Identity_2d
let m1 = max.layer (2,2) ((),())
let r = flat.layer () ((),())
let f = layers.Dense (169, 10) act.Identity_1d

let n = tf.combine c m1
let nn = tf.combine n r
let nn3 = tf.combine nn f


let main [m][d][n] (input: [m][d]tf.t) (labels: [m][n]tf.t) =
  let i = 0
  let (nnf,nnb, nnu,w) = nn3
  let (w', _) = loop (w, i) while i < 100000 do
    let input' = [unflatten 28 28 input[i]]
    let label' = labels[i:i+1]
    let (os,out) = nnf w input'
    let error = loss.softmax_cross_entropy_with_logits label' out
    let (_, g) = nnb true w os error
    let w' = nnu 0.01 w g
    in (w', i + 1)


    let acc: f32 = 0
    let i = i
    let (acc',_) = loop (acc, i) while i < 100000 do
                   let input' = [unflatten 28 28 input[i]]
                   let label' = labels[i:i+1]
                   let acc = acc + tf.accuracy (nnf, nnb, nnu, w') input' label'
                   in (acc, i+ 1)
    in acc'

    -- let input' = map (\x -> unflatten 28 28 x) input[:10]
-- in tf.accuracy (nnf, nnb, nnu, w') input' labels[:10]



   -- let (os1, os2) = os
   -- let (_, db, _, dw) = f
   -- let (derr, _) = db true dw os.2 error
   -- let (_, rb, _, rw) = r
   -- let (os1, os2) = os1 -- in (derr, length derr, length derr[0]   )
   -- let (rerr, _ ) = rb false rw os2 derr
   -- let (_, mb, _, mw) = m1
   -- let (os1, os2) = os1
   -- let (merr, _) =  mb false mw os2 rerr
   -- let (_, cb, _, cw) = c
   -- let (a, (w,b)) = cb false cw os1 merr in (w,b)