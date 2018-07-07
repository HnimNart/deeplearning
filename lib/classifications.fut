module classification (R:real) : {

  type t = R.t

  val softmax_1d: []t -> []t
  val softmax_2d: [][]t -> [][]t

} = {

  type t = R.t

  let softmax_1d_stable (X:[]t) =
    let maxval = R.maximum X
    let exps = map (\x -> R.(exp (x - maxval))) X
    let sumexps = R.(reduce (+) R.(i32 0) exps)
    in  map (\x -> R.(x / sumexps)) exps

  let softmax_2d_stable (X:[][]t) =
    map (\x -> softmax_1d_stable x) X

  let softmax_1d =
    softmax_1d_stable

  let softmax_2d =
    softmax_2d_stable

}
