import "util"
import "nn_types"

-- | Activation functions are defined as a record
--   of two function i.e.
--   1. The function, f, and
--   2. it's derivative, fd
module type activations = {

  type t

  val Identity_1d: (n: i64) -> activation_func ([n]t)
  val Sigmoid_1d:  (n: i64) -> activation_func ([n]t)
  val Relu_1d:     (n: i64) -> activation_func ([n]t)
  val Tanh_1d:     (n: i64) -> activation_func ([n]t)
  val Softmax_1d:  (n: i64) -> activation_func ([n]t)

}

module activation_funcs (R:real) : activations with t = R.t = {

  type t = R.t

  module util = utility R

  let identity_1d (X:[]t) =
    X

  let identity_1d' [m] (_:[m]t) : [m]t =
    map (\_ -> R.(i32 1)) (0..<m)

  let sigmoid_1d (X:[]t) : []t =
    map (\x -> R.(i32 1/(i32 1 + exp (negate x)))) X

  let sigmoid_1d' (X:[]t) : []t =
    map (\x -> R.(x * ((i32 1) - x))) (sigmoid_1d X)

  let relu_1d (X:[]t) :[]t =
    map (\x -> R.(max x (i32 0))) X

  let relu_1d' (X:[]t) : []t =
    map (\x -> R.(if x <= i32 0 then i32 0 else i32 1)) X

  let tanh_1d (X:[]t) : []t =
    map (\x ->
         R.((exp x - exp(negate x)) / ((exp x) + exp (negate x)))) X

  let tanh_1d' (X:[]t) : []t =
    map (\x -> R.(i32 1 - x**(i32 2))) (tanh_1d X)

  let softmax_1d_stable (X:[]t) =
    let maxval = R.maximum X
    let exps = map (\x -> R.(exp (x - maxval))) X
    let sumexps = R.sum exps
    in  map (R.((/sumexps))) exps

  -- Broken! Don't use this!
  -- Correct return val is 'retval'
  -- But is a matrix not an array
  let softmax_1d_stable' (X:[]t) =
    let softmax_res = softmax_1d_stable X
    let outer_prod  =
      map (\x -> map (\y -> R.(x * y)) softmax_res ) softmax_res
    let diagSoft    = util.diag softmax_res
    let retval      =
      map2 (\xr yr -> map2 (\x y -> R.(x - y)) xr yr ) diagSoft outer_prod
    in map (R.sum) (retval)

  --- Wrappers for activations function pairs ---
  let Identity_1d n : activation_func ([n]t) =
    {f = identity_1d, fd = identity_1d'}

  let Sigmoid_1d n : activation_func ([n]t) =
    {f = sigmoid_1d, fd = sigmoid_1d'}

  let Relu_1d n : activation_func ([n]t) =
    {f = relu_1d, fd = relu_1d'}

  let Tanh_1d n : activation_func ([n]t) =
    {f = tanh_1d, fd = tanh_1d'}

  let Softmax_1d n : activation_func ([n]t) =
    {f = softmax_1d_stable, fd = softmax_1d_stable'}

}
