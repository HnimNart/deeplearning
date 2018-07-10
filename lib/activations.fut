import "util"

module type activations = {

  type t

  type ^act_pair_1d
  val Identity_1d: act_pair_1d
  val Sigmoid_1d: act_pair_1d
  val Relu_1d: act_pair_1d
  val Tanh_1d: act_pair_1d
  val Softmax_1d: act_pair_1d

  type ^act_pair_2d
  val Identity_2d: act_pair_2d
  val Sigmoid_2d: act_pair_2d
  val Relu_2d: act_pair_2d
  val Tanh_2d: act_pair_2d
  val Softmax_2d: act_pair_2d

}

module activations_funcs (R:real) : activations with t = R.t
                                                with act_pair_1d = ([]R.t -> []R.t, []R.t -> []R.t)
                                                with act_pair_2d = ([][]R.t -> [][]R.t, [][]R.t -> [][]R.t)
                                                = {

  type t = R.t
  type act_pair_1d = ([]t -> []t, []t -> []t)
  type act_pair_2d = ([][]t -> [][]t, [][]t -> [][]t)

  module util = utility R

  let identity_1d (X:[]t) : []t  =
    X

  let identity_1d' [m] (X:[m]t) : [m]t =
    map (\_ -> R.(i32 1)) X

  let identity_2d (x:[][]t) : [][]t =
    x

  let identity_2d' (X:[][]t): [][]t =
    map (\x -> identity_1d' x) X

  let sigmoid_1d (X:[]t) : []t =
    map (\x -> R.(i32 1/(i32 1 + exp (negate x)))) X

  let sigmoid_1d' (X:[]t) : []t =
    map (\x -> R.(x * ((i32 1) - x))) (sigmoid_1d X)

  let sigmoid_2d (X:[][]t) : [][]t =
    map (\x -> sigmoid_1d x) X

  let sigmoid_2d' (X:[][]t): [][]t =
    map (\x -> sigmoid_1d' x) X

  let relu_1d (X:[]t) :[]t =
    map (\x -> R.(max x (i32 0))) X

  let relu_1d' (X:[]t) : []t =
    map (\x -> R.(if x <= i32 0 then i32 0 else i32 1)) X

  let relu_2d (X:[][]t) :[][]t =
    map (\x -> relu_1d x) X

  let relu_2d' (X:[][]t) :[][]t =
    map (\x -> relu_1d' x) X

  let tanh_1d (X:[]t) : []t =
    map (\x -> R.((exp x - exp(negate x)) / ((exp x) + exp (negate x)))) X

  let tanh_1d' (X:[]t) : []t =
    map (\x -> R.(i32 1 - x**(i32 2))) (tanh_1d X)

  let tanh_2d (X:[][]t) : [][]t =
    map (\x -> tanh_1d x) X

  let tanh_2d' (X:[][]t) : [][]t =
    map (\x -> tanh_1d' x) X

  let softmax_1d_stable (X:[]t) =
    let maxval = R.maximum X
    let exps = map (\x -> R.(exp (x - maxval))) X
    let sumexps = R.(reduce (+) R.(i32 0) exps)
    in  map (\x -> R.(x / sumexps)) exps

  let softmax_1d_stable' (X:[]t) =
    let softmax_res = softmax_1d_stable X
    let outer_prod  = map (\x -> map (\y -> R.(x * y)) softmax_res ) softmax_res
    let diagSoft    = util.diag softmax_res
    let retval      = map2 (\xr yr -> map2 (\x y -> R.(x - y)) xr yr ) diagSoft outer_prod
    in util.extract_diag retval

  let softmax_2d_stable (X:[][]t) =
    map (\x -> softmax_1d_stable x) X

  let softmax_2d_stable' (X:[][]t) =
    map (\x -> softmax_1d_stable' x) X




----- Collections --------
  let Identity_1d  =
    (identity_1d, identity_1d')

  let Identity_2d  =
    (identity_2d, identity_2d')

  let Sigmoid_1d =
    (sigmoid_1d, sigmoid_1d')

  let Sigmoid_2d =
    (sigmoid_2d, sigmoid_2d')

  let Relu_1d =
    (relu_1d, relu_1d')

  let Relu_2d =
    (relu_2d, relu_2d')

  let Tanh_1d =
    (tanh_1d, tanh_1d')

  let Tanh_2d =
    (tanh_2d, tanh_2d')

  let Softmax_1d =
    (softmax_1d_stable, softmax_1d_stable')

  let Softmax_2d =
    (softmax_2d_stable, softmax_2d_stable')

}
