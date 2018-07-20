import "util"
import "nn_types"

-- | Activation functions are defined as a tuple
--   of two function i.e.
--   1. The function itself and
--   2. it's derivative
module type activations = {

  type t

  type ^act_pair_1d
  val Identity_1d: act_pair_1d
  val Sigmoid_1d: act_pair_1d
  val Relu_1d: act_pair_1d
  val Tanh_1d: act_pair_1d
  val Softmax_1d: act_pair_1d

}

module activation_funcs (R:real) : activations with t = R.t
                                               with act_pair_1d = f_pair_1d R.t = {
  type t = R.t
  type act_pair_1d = f_pair_1d t

  module util = utility R

  let identity_1d (X:[]t) : []t  =
    X

  let identity_1d' [m] (X:[m]t) : [m]t =
    map (\_ -> R.(i32 1)) X

  let sigmoid_1d (X:[]t) : []t =
    map (\x -> R.(i32 1/(i32 1 + exp (negate x)))) X

  let sigmoid_1d' (X:[]t) : []t =
    map (\x -> R.(x * ((i32 1) - x))) (sigmoid_1d X)

  let relu_1d (X:[]t) :[]t =
    map (\x -> R.(max x (i32 0))) X

  let relu_1d' (X:[]t) : []t =
    map (\x -> R.(if x <= i32 0 then i32 0 else i32 1)) X

  let tanh_1d (X:[]t) : []t =
    map (\x -> R.((exp x - exp(negate x)) / ((exp x) + exp (negate x)))) X

  let tanh_1d' (X:[]t) : []t =
    map (\x -> R.(i32 1 - x**(i32 2))) (tanh_1d X)

  let softmax_1d_stable (X:[]t) =
    let maxval = R.maximum X
    let exps = map (\x -> R.(exp (x - maxval))) X
    let sumexps = R.(reduce (+) R.(i32 0) exps)
    in  map (\x -> R.(x / sumexps)) exps

  let softmax_1d_stable' (X:[]t) =
    let softmax_res = softmax_1d_stable X
    let outer_prod  = map (\x -> map (\y -> R.(x * y)) softmax_res ) softmax_res
    let diagSoft    = util.diag softmax_res
    let matrix      = map2 (\xr yr -> map2 (\x y -> R.(x - y)) xr yr ) diagSoft outer_prod
    in  map (R.sum) (matrix)


----- Collections --------
  let Identity_1d  =
    (identity_1d, identity_1d')


  let Sigmoid_1d =
    (sigmoid_1d, sigmoid_1d')


  let Relu_1d =
    (relu_1d, relu_1d')

  let Tanh_1d =
    (tanh_1d, tanh_1d')


  let Softmax_1d =
    (softmax_1d_stable, softmax_1d_stable')


}
