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

let conv1 = conv2d.layer (32, 5) act.Identity_2d
let max_pooling1 = max.layer (2,2) ((), ())
let conv2 = conv2d.layer (64 ,3) act.Identity_2d
let max_pooling2 = max.layer (2,2) ((),())
let flat2 = flat.layer () ((),())
let dense = layers.Dense (1600, 1024) act.Identity_1d
let output = layers.Dense (1024, 10) act.Identity_1d


let nn1 = tf.combine conv1 max_pooling1
let nn2 = tf.combine nn1 conv2
let nn3 = tf.combine nn2 max_pooling2
let nn4 = tf.combine nn3 flat2
let nn5 = tf.combine nn4 dense
let nn  = tf.combine nn5 output


let get_dims (X:[][][]f32) =
  (length X, length X[0], length X[0,0])

let main [m][d][n] (input: [m][d]tf.t) (labels: [m][n]tf.t) =
  let i = 0
  let (nnf,nnb, nnu,w) = nn
  let input' = [unflatten 28 28 input[i]]
  let label' = labels[i:i+1]
  let (os, out) = nnf w input'
  let error = loss.softmax_cross_entropy_with_logits label' out
  let (os1, os2) = os
  let (_, db, _, dw) = output
  let (derr, _) = db true dw os.2 error
  let (_, rb, _, rw) = dense
  let (os1, os2) = os1
  let (rerr, _ ) = rb false rw os2 derr
  let (_, mb, _, mw) = flat2
  let (os1, os2) = os1
  let (merr, _) =  mb false mw os2 rerr
  let (_, cb, _, cw) = max_pooling2
  let (os1, os2) = os1
  let (p2err, _) = cb false cw os2 merr
  let (_, c2b, _, c2w) = conv2
  let (os1, os2) = os1
  let (c2err, (w,b)) = c2b false c2w os2 p2err in (get_dims c2err)
  -- let (_, p1b, _, p1w) = max_pooling1
  -- let (os1, os2) = os1
  -- let (p1err, _) = p1b false p1w os2 c2err in (get_dims c2err, c2err)



   --in (if f32.isnan out[0, 6] then 1 else 0, f32. maximum (intrinsics.flatten dw.1))

  -- let (w', _) = loop (w, i) while i < 100000 - 2 do
  --   let input' = [unflatten 28 28 input[i]]
  --   let label' = labels[i:i+1]
  --   let (os,out) = nnf w input'
  --   let error = loss.softmax_cross_entropy_with_logits label' out
  --   let (_, g) = nnb true w os error
  --   let w' = nnu 0.01 w g
  --   in (w', i + 1)


  --   let acc: f32 = 0
  --   let i = i
  --   let (acc',_) = loop (acc, i) while i < 100000 - 2 do
  --                  let input' = [unflatten 28 28 input[i]]
  --                  let label' = labels[i:i+1]
  --                  let acc = acc + tf.accuracy (nnf, nnb, nnu, w') input' label'
  --                  in (acc, i+ 1)
  --   in acc'

    -- let input' = map (\x -> unflatten 28 28 x) input[:10]
-- in tf.accuracy (nnf, nnb, nnu, w') input' labels[:10]

let main2 [m][d][n] (input: [m][d]tf.t) (labels: [m][n]tf.t) =
  let i = 0
  let (nnf,nnb, nnu,w) = nn
    -- let input' = [unflatten 28 28 input[i]]
    -- let label' = labels[i:i+1]
    -- let (os, out) = nnf w input' in out
  let (w', _) = loop (w, i) while i < 1000 - 2 do
    let input' = [unflatten 28 28 input[i]]
    let label' = labels[i:i+1]
    let (os,out) = nnf w input'
    let error = loss.softmax_cross_entropy_with_logits label' out
    let (_, g) = nnb true w os error
    let w' = nnu 0.01 w g
    in (w', i + 1)


    let acc: f32 = 0
    let i = i
    let (acc',_) = loop (acc, i) while i < 1000 - 2 do
                   let input' = [unflatten 28 28 input[i]]
                   let label' = labels[i:i+1]
                   let acc = acc + tf.accuracy (nnf, nnb, nnu, w') input' label'
                   in (acc, i+ 1)
    in acc'

-- let c = conv2d.layer (3, 3) act.Identity_2d
-- let m1 = max.layer (2,2) ((),())
-- let r = flat.layer () ((),())
-- let f = layers.Dense (169*3, 10) act.Identity_1d

-- let n = tf.combine c m1
-- let nn = tf.combine n r
-- let nn3 = tf.combine nn f
