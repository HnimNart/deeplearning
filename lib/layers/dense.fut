import "layer_type"
import "../nn_types"
import "../util"
import "../random_gen"
import "/futlib/linalg"

module dense (R:real) : layer with t = R.t
                              with input_params = (i32, i32)
                              with activations = (f_pair_1d R.t)
                              with layer = dense_tp R.t = {

  type t            = R.t
  type input        = arr2d t
  type weights      = std_weights R.t
  type output       = arr2d t
  type garbage      = (arr2d t, arr2d t)
  type error_in     = arr2d t
  type error_out    = arr2d t
  type gradients    = (error_out, weights)
  type input_params = (i32, i32)
  type activations  = f_pair_1d t

  type layer = dense_tp t

  module lalg   = linalg R
  module util   = utility R
  module random = normal_random_array R

  let empty_garbage:garbage= ([[]],[[]])

  ---- Each input is in row
  let forward  (act:[]t -> []t) (training:bool) ((w,b):weights) (input:input) : (garbage, output) =
    let res      = lalg.matmul w (transpose input)
    let res_bias = map2 (\xr b -> map (\x -> (R.(x + b))) xr) res b
    let res_act  = map (\x -> act x) res_bias
    let garbage  = if training then (input, res_bias) else empty_garbage
   in (garbage, transpose res_act)

  let backward (act:[]t -> []t) ((w,_):weights) ((input, inp_w_bias):garbage) (error:error_in)  =
    let (res_m, res_n)   = (length inp_w_bias, length inp_w_bias[0])
    let deriv            = unflatten res_m res_n (act (flatten inp_w_bias))
    let delta            = util.mult_matrix (transpose error) deriv
    let w_grad           = lalg.matmul delta (input)
    let b_grad           = map (R.sum) delta
    let error'           = transpose (lalg.matmul (transpose w) delta)
   in (error', (w_grad, b_grad))

  let update (f:apply_grad t) (w: weights) (wg:weights)  =
    f w wg

  let init ((m,n):input_params) (act:activations) (seed:i32)  =
    let w = random.gen_random_array_2d_w_scaling (m,n) seed
    let b = (map (\_ -> R.(i32 0)) (0..<n))
    in
    (forward act.1,
     backward act.2,
     update,
    (w,b))

}
