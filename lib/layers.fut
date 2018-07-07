import "types"
import "activations"
import "/futlib/linalg"
import "util"


module type layer = {

  type t
  type input
  type weights
  type output
  type error_in
  type error_out
  type gradients

  type act
  type layer
  type input_params

  -- val forward: act -> weights -> input -> output
  -- val backward:  act -> bool ->  weights ->  input -> error_in -> gradients
  val layer: input_params -> (act, act) -> layer

  -- val get_ws: layer -> weights
  -- val get_f: layer -> weights -> input -> (input, output)
  -- val get_b: layer -> bool -> weights ->  input -> error_in -> gradients
}

module dense (R:real) : layer with t = R.t
                              with input = [][]R.t
                              with weights = ([][]R.t, [][]R.t)
                              with input_params = (i32, i32)
                              with output  = ([][]R.t)
                              with error_in = ([][]R.t)
                              with error_out = ([][]R.t)
                              with gradients = ([][]R.t ,([][]R.t, [][]R.t))
                              with layer = NN ([][]R.t) ([][]R.t,[][]R.t) ([][]R.t) ([][]R.t) ([][]R.t) ([][]R.t) (R.t)
                              with act = ([]R.t -> []R.t) = {

  type t = R.t
  type input = [][]t
  type weights = ([][]t, [][]t)
  type output = [][]t
  type garbage = [][]t
  type error_in = [][]t
  type error_out = [][]t
  type gradients = (error_out, weights)
  type input_params = (i32, i32)

  type act = []t -> []t
  type layer = NN input weights output garbage error_in error_out t

  module lalg   = linalg R
  module util   = utility_funcs R
  module random = normal_random_array R

   ---- Each input is in a row
  let forward  (act:act) ((w,b):weights) (input:input) : output =
    let product = lalg.matmul w (transpose input)
    let product' = map2 (\xr b -> map (\x -> (R.(x + b[0]))) xr) product b
    let (m, k) = (length product', length product'[0])
    let output = act (flatten product')
   in transpose (unflatten m k output)

  let backward (act:act) (l_layer:bool) ((w,_):weights) (input:input) (error:error_in)  =
    if l_layer then
      let error_corrected = (map (map R.((/i32 (length input) ))) (transpose error))
      let w_grad          = lalg.matmul (error_corrected) (input)
      let b_grad          = transpose  [map (R.sum) error_corrected]
      let error'          = lalg.matmul (transpose w) error_corrected
      in (error', (w_grad, b_grad))
    else
      let res            = lalg.matmul (w) (transpose input) -- laeg bias til
      let (res_m, res_n) = (length res, length res[0])
      let deriv          = unflatten res_m res_n (act (flatten res))
      let delta          = util.multMatrix error deriv
      let w_grad           = lalg.matmul delta (input)
      let b_grad         = transpose [map (R.sum) delta]
      let error'         = lalg.matmul (transpose w) delta
      in (error', (w_grad, b_grad))

  let update (alpha:t) ((w,b): weights) ((wg,bg):weights)  =
    let bg_scaled        = util.scaleMatrix bg alpha
    -- let wg' = map (\xr -> map (\x -> R.(max (i32 1) (min (negate (i32 1)) x ))) xr) wg
    let wg'  = map (\xr -> map (\x -> R.(if x > i32 1 then i32 1 else if x < (negate (i32 1)) then (negate (i32 1)) else x)) xr ) wg
    let wg_scaled        = util.scaleMatrix wg' alpha
    let w'               = util.subMatrix w wg_scaled
    let b'               = util.subMatrix b bg_scaled
    in (w', b')


  let layer ((m,n):input_params) (act_id: (act, act))   =
    let w = random.gen_random_array_2d (m,n) 1
    let b = unflatten n 1 (map (\_ -> R.( i32 0)) (0..<n))
    in
    (\w input -> (input, forward act_id.1 w input),
     (backward act_id.2),
      update,
     (w,b))

  let get_f (nn:layer) = nn.1
  let get_b (nn:layer) = nn.2
  let get_ws (nn:layer): weights = nn.4
}


module layers (R:real) :{

  type t = R.t
  type dense_tp = NN ([][]t) ([][]t, [][]t) ([][]t) ([][]t) ([][]t) ([][]t) t
  module dense : layer with t = R.t
                       with input = [][]R.t
                       with weights = ([][]R.t, [][]R.t)
                       with output = [][]R.t
                       with error_in = ([][]R.t)
                       with error_out = ([][]R.t)
                       with gradients = ([][]R.t, ([][]R.t, [][]R.t))
                       with act = ([]R.t -> []R.t)
                       with layer = dense_tp

   val Dense: (i32, i32) -> (dense.act, dense.act) -> dense.layer

} = {

  type t = R.t
  type dense_tp = NN ([][]t) ([][]t, [][]t) ([][]t) ([][]t) ([][]t) ([][]t) t
  module dense = dense R


  let Dense ((m,n):(i32,i32)) (act_id: (dense.act, dense.act))  =
      dense.layer (m,n) act_id

  -- type act_pair_1d = i32
  -- type dense = NN

}
