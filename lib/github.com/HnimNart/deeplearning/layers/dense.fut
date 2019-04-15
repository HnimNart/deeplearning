import "layer_type"
import "../nn_types"
import "../util"
import "../weight_init"
import "../../../diku-dk/linalg/linalg"


-- | Fully connected layer
module dense (R:real) : layer_type with t = R.t
                                   with input_params = (i32, i32)
                                   with activations  = activation_func ([]R.t)
                                   with input        = arr2d R.t
                                   with weights      = std_weights R.t
                                   with output       = arr2d R.t
                                   with cache        = (arr2d R.t, arr2d R.t)
                                   with error_in     = arr2d R.t
                                   with error_out    = arr2d R.t = {

  type t            = R.t
  type input        = arr2d t
  type weights      = std_weights t
  type output       = arr2d t
  type cache        = (arr2d t, arr2d t)
  type error_in     = arr2d t
  type error_out    = arr2d t
  type b_output     = (error_out, weights)

  type input_params = (i32, i32)
  type activations  = activation_func ([]t)

  type dense_tp = NN input weights output
                     cache error_in error_out (apply_grad t)

  module lalg   = mk_linalg R
  module util   = utility R
  module w_init = weight_initializer R

  let empty_cache:cache= ([[]],[[]])
  let empty_error:error_out = [[]]

  -- Forward propagation
  let forward  (act:[]t -> []t)
               (training:bool)
               ((w,b):weights)
               (input:input) : (cache, output) =

    let res      = lalg.matmul w (transpose input)
    let res_bias = transpose (map2 (\xr b' -> map (\x -> (R.(x + b'))) xr) res b)
    let res_act  = map (\x -> act x) (res_bias)
    let cache    = if training then (input, res_bias) else empty_cache
    in (cache, res_act)

  -- Backward propagation
  let backward (act:[]t -> []t)
               (first_layer:bool)
               (apply_grads:apply_grad t)
               ((w,b):weights)
               ((input, inp_w_bias):cache)
               (error:error_in) : b_output =

    let deriv    = (map (\x -> act x) inp_w_bias)
    let delta    = transpose (util.hadamard_prod_2d error deriv)
    let w_grad   = lalg.matmul delta input
    let b_grad   = map (R.sum) delta
    let (w', b') = apply_grads (w,b) (w_grad, b_grad)

    --- Calc error to backprop to previous layer
    let error' =
      if first_layer
      then
       copy empty_error
      else
       transpose (lalg.matmul (transpose w) delta)
    in (error', (w', b'))


  let init ((m,n):input_params) (act:activations) (seed:i32) : dense_tp =
    let w = w_init.gen_random_array_2d_xavier_uni (m,n) seed
    let b = map (\_ -> R.(i32 0)) (0..<n)
    in
    {forward  = forward act.f,
     backward = backward act.fd,
     weights  = (w,b)}

}
