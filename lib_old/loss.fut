import "activations"
import "classifications"

--tf.nn.sigmoid_cross_entropy_with_logits
--tf.nn.softmax ?
--tf.nn.log_softmax
--tf.nn.softmax_cross_entropy_with_logits
--tf.nn.softmax_cross_entropy_with_logits_v2 -
--tf.nn.sparse_softmax_cross_entropy_with_logits
--tf.nn.weighted_cross_entropy_with_logits


module type loss = {

  type t
  val loss : []t -> []t -> t
  val derivative: []t -> []t -> []t
}


module cross_entropy(R:real) : loss with t = R.t = {

  type t = R.t

  let loss [d] (labels: [d]R.t) (logits: [d]R.t) =
    let x = map2 (\x y -> if R.(isinf (log y)) then R.(i32 0) else R.((log y) * x)) labels logits
    in R.(negate (reduce (\x y -> R.(x + y)) R.(i32 0) x))

  let derivative [d] (target: [d]R.t) (estimation: [d]R.t) =
    map2 (\x y -> R.((negate (i32 1 )) * (y * (i32 1/x) + (i32 1 - y) * (i32 1/(i32 1 - x))))) target estimation

}


module softmax_cross_entropy_with_logits (R:real) : loss with t = R.t = {

  type t = R.t
  module softmax = softmax_stable R
  module cross_entro = cross_entropy R

  let loss [d] (target: [d]R.t) (estimation: [d]t) =
    let softmax_res = softmax.classify estimation
    in (cross_entro.loss target softmax_res)

  let derivative [d] (labels: [d]R.t) (logits:[d]R.t) =
    -- map (\_ -> R.(i32 10)) (0..<length target)
     map2 (\x y -> R.(y - x)) labels (softmax.classify logits)
}


--- SE (Squared error) ----
module squared_error (R:real) : loss with t = R.t = {

  type t = R.t
  let abs (x: R.t): R.t = R.(if x < i32 0 then negate x else x)

  let squaredDiff ( x:R.t ) (y:R.t) =
    let diff = R.(x - y)
    let absdiff = abs diff
    in R.(absdiff**(i32 2))

   let loss [d] (target: [d]R.t) (estimation: [d]R.t) =
     reduce (\x y -> R.(x + y)) R.(i32 0) (map2 (\t y -> R.(squaredDiff t y) ) target estimation)

   let derivative [d] (target: [d]R.t) (estimation: [d]R.t) =
    map2 (\x y -> R.(x -y)) estimation target
}




module type loss_funcs = {
  type t
  val calc_loss_deriv: []t -> []t -> i32 -> []t
  val calc_loss: []t -> []t -> i32 -> t
}

module loss_funcs (R:real) : loss_funcs with t = R.t = {

  type t = R.t

  module soft_cross = softmax_cross_entropy_with_logits R
  module MSE = squared_error R

  let calc_loss [m] (labels:[m]t) (logits:[m]t) (loss_id: i32) =
    if loss_id  == 1 then soft_cross.loss labels logits
    else if loss_id == 2 then MSE.loss labels logits
    else R.(i32 0)

  let calc_loss_deriv [m] (labels: [m]t) (logits:[m]t) (loss_id: i32) =
    if loss_id == 1 then soft_cross.derivative labels logits
    else if loss_id == 2 then MSE.derivative labels logits
    else replicate m R.(i32 0)


}