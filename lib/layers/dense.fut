import "layer_type"
import "../nn_types"
import "../activations"
import "/futlib/linalg"
import "../util"

module dense (R:real) : layer with t = R.t
                              with input = [][]R.t
                              with weights = ([][]R.t, []R.t)
                              with input_params = (i32, i32)
                              with output  = ([][]R.t)
                              with error_in = ([][]R.t)
                              with error_out = ([][]R.t)
                              with gradients = ([][]R.t ,([][]R.t, []R.t))
                              with layer = NN ([][]R.t) ([][]R.t,[]R.t) ([][]R.t) ([][]R.t) ([][]R.t) ([][]R.t) (updater ([][]R.t, []R.t))
                              with act = ([]R.t -> []R.t) = {

  type t = R.t
  type input = [][]t
  type weights = ([][]t, []t)
  type output = [][]t
  type garbage = [][]t
  type error_in = [][]t
  type error_out = [][]t
  type gradients = (error_out, weights)
  type input_params = (i32, i32)

  type act = []t -> []t
  type layer = NN input weights output garbage error_in error_out (updater weights)

  module lalg   = linalg R
  module util   = utility R
  module random = normal_random_array R

  let empty_garbage:garbage= [[]]
   ---- Each input is in a row
  let forward  (act:act) (training:bool) ((w,b):weights) (input:input) : (garbage, output) =
    let product = lalg.matmul w (transpose input)
    let product' =  map2 (\xr b -> map (\x -> (R.(x + b))) xr) product b
    let (m, k) = (length product', length product'[0])
    let output = act (flatten product')
    let garbage = if training then input else  empty_garbage
   in (garbage, transpose (unflatten m k output))

  let backward (act:act) ((w,b):weights) (input:input) (error:error_in)  =
    let res              = lalg.matmul (w) (transpose input)
    let res_bias         = map2 (\xr b -> map (\x -> (R.(x + b))) xr) res b
    let (res_m, res_n)   = (length res_bias, length res_bias[0])
    let deriv            = unflatten res_m res_n (act (flatten res_bias))
    let delta            = util.mult_matrix error deriv
    let w_grad           = lalg.matmul delta (input)
    let b_grad           = map (R.sum) delta
    let error'           = lalg.matmul (transpose w) delta
    in (error', (w_grad, b_grad))

  let update (f:updater weights) (w: weights) (wg:weights)  =
    f w wg

  let init ((m,n):input_params) (act_id: (act, act)) (seed:i32)  =
    let w = random.gen_random_array_2d_w_scaling (m,n) seed
    let b = (map (\_ -> R.(i32 0)) (0..<n))
    in
    (forward act_id.1 ,
     backward act_id.2,
     update,
     (w,b))

}
