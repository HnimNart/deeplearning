import "util"
import "activations"
import "/futlib/linalg"
import "classifications"

type forwards 'input 'w 'output  'garbage     = w -> input -> (garbage, output) --- NN
type train 'input 'w 'delta  'new_w 'output   = w -> input -> input ->  delta -> (delta, new_w)

-- type NN 'input 'w 'labels 'output 'delta 'new_w      = (forwards input w output, train input w  delta new_w output, w)
type NN 'input 'w 'output 'garbage = (forwards input w output garbage, w)

   ---- input weights labels output delta new_weights
type dense 't = NN ([][]t) ([][]t, [][]t) ([][]t) ([][]t)
type loss  't = NN ([][]t)  () ([][]t) ([][]t)


module random = normal_random_array f32
module activation = activation_funcs_coll f32
module lalg   = linalg f32
module cross = softmax_cross_entropy_with_logits f32
module util  = utility_funcs f32

let dense_forward [m][n][k] (act_id:i32) ((w,b):([m][n]f32, [m][1]f32)) (input:[n][k]f32) : ([m][k]f32) =
  let product = lalg.matmul w input
  let product' = map2 (\xr yr -> map2 (\x y -> (x + y)) xr yr) product b
  let act = activation.calc_activation (flatten product') act_id
  in unflatten m k act

let dense_train [m][n] (act_id:i32) ((w,b):([m][n]f32, [m][1]f32)) (input1:[][]f32) (input2:[][]f32) (delta:[][]f32)   =
  let error = lalg.matmul (transpose w) delta
  let (input_m, input_n) = (length input2, length input2[0])
  let deriv    = unflatten input_m input_n (activation.calc_derivative (flatten input2) act_id)
  let delta    = util.multMatrix error deriv
  let grads    = lalg.matmul (delta) (transpose input1)
  let grads_w' = util.scaleMatrix grads 0.01
  let w'       = util.subMatrix w grads_w'
  let b'       = util.subMatrix b delta
  in (delta, (w',b'))

let dense ((m,n):(i32,i32)) (act_id: i32) : dense f32 =
  let w = unflatten n m (random.gen_random_array (m*n) 1)
  let b = unflatten n 1 (random.gen_random_array (n) 1)
  in (\w input -> (input, dense_forward act_id w input), (w,b))

let combine 'w1 'w2 'i1 'o1 'o2 'g1 'g2 ((f1,ws1): NN i1 w1 o1 g1) ((f2,ws2):NN o1 w2 o2 g2) : NN i1 (w1,w2) (o2) (g1,g2) =
  ((\(w1, w2) (input) ->  let (g1, res)  = f1 w1 input
                          let (g2, res2) = f2 w2 res
                          in ((g1, g2), res2)),
   (ws1, ws2))

let train 'w 'i 'o 'g ((f,w):NN i w o g) (input:i) (labels:o) (loss_id:i32) (alpha:f32) =
let (os, output): (g, o) = f w input in output


let n = 100
let input  = unflatten n 784 (random.gen_random_array (784*n) 1)
let labels = unflatten n 10 (random.gen_random_array (10*n) 1)
-- let w = unflatten 256 784 (random.gen_random_array (784*256) 1)
-- let b = unflatten 256 1 (random.gen_random_array 256 1)

let main = let l1 = dense (784, 256) 0
           let l2 = dense (256, 128) 0
           let l3 = dense (128, 10) 0
           -- let loss: loss = loss
           let nn  = combine l1 l2
           let nn1 = combine nn l3 in
           map2 (\x y -> train nn1 (transpose [x]) (transpose [y]) (1i32) (0.01f32)) input labels
