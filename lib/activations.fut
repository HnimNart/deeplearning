



module activations (R:real) : {

  type t = R.t

  type act_pair_1d = ([]t -> []t, []t-> []t)
  val Identity_1d: act_pair_1d
  val Sigmoid_1d: act_pair_1d
  val Relu_1d: act_pair_1d
  val Tanh_1d: act_pair_1d

  type act_pair_2d =  ([][]t -> [][]t, [][]t -> [][]t)
  val Identity_2d: act_pair_2d
  val Sigmoid_2d: act_pair_2d
  val Relu_2d: act_pair_2d
  val Tanh_2d: act_pair_2d


} = {

  type t = R.t

  type act_pair_1d = ([]t -> []t, []t -> []t)
  type act_pair_2d = ([][]t -> [][]t, [][]t -> [][]t)

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
    map (\x -> R.(if x <= i32 0 then i32 0 else x)) X

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



}
