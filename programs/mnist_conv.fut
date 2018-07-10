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

module tf = network f32
module layers = layers f32
module act = activations f32
module sgd = sgd f32
module conv2d = conv2d f32
module max = max_pooling_2d f32
module flat = flatten f32
module loss = loss f32
module class = classification f32
module lalg = linalg f32


let conv1 = conv2d.layer (32, 5, 1) act.Relu_1d
let max_pooling1 = max.layer (2,2) ((), ())
let conv2 = conv2d.layer (64, 3, 1) act.Relu_1d
let max_pooling2 = max.layer (2,2) ((), ())
let flat2 = flat.layer () ((),())
-- let dense = layers.Dense (1600, 1024) act.Identity_1d
-- let dense = layers.Dense (5408, 1024) act.Identity_1d
let loutput= layers.Dense (1600, 10) act.Identity_1d

let nn1 = tf.combine conv1 max_pooling1
let nn2 = tf.combine nn1 conv2
let nn3 = tf.combine nn2 max_pooling2

-- let nn3 = tf.combine conv2 max_pooling2
let nn4 = tf.combine nn3 flat2
-- let nn5 = tf.combine nn4 dense
let nn = tf.combine nn4 loutput

let get_dims (X:[][][][]f32) =
  (length X, length X[0], length X[0,0], length X[0,0,0])

let main [m][d][n] (input: [m][d]tf.t) (labels: [m][n]tf.t) =
  let i = 00
  let n = 100
  let batch_size = 100
  let tmp = map (\img -> [unflatten 28 28 img]) input
  let (nnf,nnb, nnu,w) = nn

  let j = 1
  let (w', _) = loop (w, j) while j < 100 do
    let (w', _) = loop (w, i) while i < n do
      let input' = tmp[i:i+batch_size]
      let label' = labels[i:i+batch_size]
      let (os,out) = nnf w input'
      let error = loss.softmax_cross_entropy_with_logits label' out
      let (_, g) = nnb true w os error
      let w' = nnu (0.0005) w g
    in (w', i + batch_size)
  in (w' , j + 1)

  in tf.accuracy (nnf, nnb, nnu , w') tmp[:n] labels[:n]

  -- let acc = 0.0
  -- let j = 0
  -- let datasets = 10000
  -- let offset = 100
  -- let (acc, _) = loop (acc, j) while j < datasets do
  --             let acc = acc +  (tf.accuracy (nnf, nnb, nnu, w') tmp[j:j+offset] labels[j:j+offset])
  --             in (acc, j + offset)
  -- in acc/ f32.(i32 (datasets))

-- let (os, output) = nnf w tmp[i:i+batch_size]

  --   let error = loss.softmax_cross_entropy_with_logits label' output
  -- let (_, ob, _, ow) = loutput
  -- let (os1, os2) = os
  -- let (oerr, _) = ob true ow os2 error
  -- let (_, db, _, dw) = dense
  -- let (os1, os2) = os1
  -- let (derr, _)  = db false dw os2 oerr
  -- let (_, fb, _, fw) = flat2
  -- let (os1, os2) = os1
  -- let (ferr, _) = fb false fw os2 derr
  -- let (_, mb2, _, mw2) = max_pooling2
  -- let (os1, os2) = os1
  -- let (merr2, _) = mb2 false mw2 os2 ferr
  -- let (_, cb2, _, cw2) =  conv2
  -- let (os1, os2) = os1
  -- let (c2err, (wc, _)) = cb2 false cw2 os2 merr2 in (length wc, length wc[0])
  -- let (_, mb1, _, mw1) = max_pooling1
  -- let (os1, os2) = os1
  -- let (merr1, _) = mb1 false mw1 os2 c2err
  -- let (_, c1, _, cw1) = conv1
  -- -- let (os1, os2) = os1
  -- let (cerr1, _) = c1 false cw1 os1 merr1 in get_dims cerr1
