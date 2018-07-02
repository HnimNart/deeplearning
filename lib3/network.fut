import "util"
import "activations"
import "/futlib/linalg"
import "classifications"

type forwards 'input 'w 'output                      = w -> input -> output --- NN
type train 'input 'w 'delta  'new_w 'output          = w -> input ->  delta -> (delta, new_w)
type NN 'input 'w 'labels 'output 'delta 'new_w      = (forwards input w output, train input w  delta new_w output, w)


---- input weights labels output delta new_weights
type dense = NN ([][]f32) ([][]f32, [][]f32) ([][]f32) ([][]f32) ([][]f32) ([][]f32,[][]f32)
type loss  = NN ([][]f32)  () ([][]f32)  ([][]f32) ([][]f32) (())


module random = normal_random_array f32
module activation = activation_funcs_coll f32
module lalg   = linalg f32
module cross = softmax_cross_entropy_with_logits f32
module util  = utility_funcs f32

let dense_forward [m][n][k] (act_id:i32) ((w,b):([m][n]f32, [m][1]f32)) (input:[n][k]f32) : [m][k]f32 =
  let product = lalg.matmul w input
  let product' = map2 (\xr yr -> map2 (\x y -> f32.(x + y)) xr yr) product b
  let act = activation.calc_activation (flatten product') act_id
  in (unflatten m k act)

let dense_train [m][n] (act_id:i32) ((w,b):([m][n]f32, [m][1]f32))  (input:[][]f32)  (delta:[][]f32)   =
  -- let res = dense_forward act_id (w,b) input in (res, (w,b))
  let error = lalg.matmul (transpose w) delta
  let (input_m, input_n) = (length input, length input[0])
  let deriv = unflatten input_m input_n (activation_funcs_coll.calc_derivative (flatten input) act_id)
  let delta  = util.multMatrix error deriv
  in (delta, (w,b))

let dense ((m,n):(i32,i32)) (act_id: i32) : dense =
  let w = unflatten n m (random.gen_random_array (m*n) 1)
  let b = unflatten n 1 (random.gen_random_array (n) 1)
  in (dense_forward act_id, dense_train act_id, (w,b))


let loss :loss =
  -- let func = (\ () (labels,input) ->  cross.derivative (labels) (input))
  (\() x -> x, \() input labels -> (input,()), ())

let combine 'w1 'w2 'i1  'o1 'o2 'd1 'd2 'l1 'w1' 'w2' ((f1,b1,ws1): NN i1 w1 l1 o1 d1 w1') ((f2,b2,ws2):NN o1 w2 l1 o2 d2 w2') : NN i1 (w1,w2) (l1) (o2) (d1) (w1') =
  ((\(w1, w2) (input:i1) -> let res:o1 = f1 w1 input in f2 w2 res),
   (\(w1,w2) (input:i1) (labels:l1)  -> let res1:o1 = f1 w1 input
                                        let res2:o2 = f2 w2 res1
                                        let (delta, w2') = b2 w2 res2 []

    b1 w1 input labels delta),
   (ws1, ws2))

let combine_loss

let input = unflatten 784 1 (random.gen_random_array (784) 1)
let labels = unflatten 10 1 (random.gen_random_array 10 1)
-- let w = unflatten 256 784 (random.gen_random_array (784*256) 1)
-- let b = unflatten 256 1 (random.gen_random_array 256 1)
-- let main = dense_forward 0 (w,b) input

let main = let l1: dense = dense (784, 256) 0
           let l2: dense = dense (256, 256) 0
           let l3: dense = dense (256, 10) 0
           let loss: loss = loss
           let nn = combine l1 l2
           let (nn1) = combine nn l3
           let (_, train, w') = combine nn1 loss
           let (a,b) = (train w' input labels)
            in length a


-- ((w1, w2) -> i1 -> o2, (w1, w2) -> i1 -> l1 -> (d2, w2'), (w1, w2))
-- ((w1, w2) -> i1 -> o2, (w1, t3) -> i1 -> l1 -> (d1, w1'), (w1, w2))
