import "activations_funcs"

-- | loss functions are defined as a tuple
--   of two function i.e.
--   1. The function itself and
--   2. it's derivative w.r.t. to logits
module type loss = {

  type t
  type ^loss_1d
  type ^loss_2d

  type ^loss_deriv_1d
  type ^loss_deriv_2d

  val cross_entropy : (loss_1d, loss_deriv_1d)
  val softmax_cross_entropy_with_logits : (loss_1d, loss_deriv_1d)
  val softmax_cross_entropy_with_logits_2d : (loss_2d, loss_deriv_2d)
}

module loss_funcs (R:real) : loss with t = R.t
                                  with loss_1d = ([]R.t -> []R.t -> R.t)
                                  with loss_2d = ([][]R.t -> [][]R.t -> R.t)
                                  with loss_deriv_1d = ([]R.t -> []R.t -> []R.t)
                                  with loss_deriv_2d = ([][]R.t -> [][]R.t -> [][]R.t) = {

  type t = R.t
  type loss_1d = []t -> []t -> t
  type loss_2d = [][]t -> [][]t -> t

  type loss_deriv_1d = []t -> []t -> []t
  type loss_deriv_2d = [][]t -> [][]t -> [][]t

  module activations = activation_funcs R

  let cross_entropy_1d [d] (logits:[d]t) (labels:[d]t) =
    let res = map2 (\x y -> if R.(isinf (log y)) then R.(i32 0) else R.((log y) * x)) labels logits
    in R.(negate (reduce (\x y -> R.(x + y)) R.(i32 0) res))

  let cross_entropy_1d' [d] (logits:[d]t) (labels:[d]t) =
    map2 (\x y -> R.((negate (i32 1 )) * (y * (i32 1/x) + (i32 1 - y) * (i32 1/(i32 1 - x))))) labels logits

  let softmax_cross_entropy_with_logits_stable_1d [d] (logits:[d]t) (labels:[d]t) =
    let softmax_res = activations.Softmax_1d.1 logits
    in cross_entropy_1d softmax_res labels

  let softmax_cross_entropy_with_logits_stable_1d' [d] (logits:[d]t) (labels:[d]t) =
    let softmax_res = activations.Softmax_1d.1 logits
    in map2 (\x y -> R.(y - x)) labels softmax_res

  let softmax_cross_entropy_with_logits_stable_2d [d] (logits: [d][]R.t) (labels:[d][]R.t) =
    let loss =  map2 (\x y -> softmax_cross_entropy_with_logits_stable_1d  x y) logits labels
    in reduce (R.+) R.(i32 0) loss

  let softmax_cross_entropy_with_logits_stable_2d' [d] (logits: [d][]R.t) (labels:[d][]R.t) =
     map2 (\x y -> softmax_cross_entropy_with_logits_stable_1d' x y) logits labels

  let softmax_cross_entropy_with_logits =
    (softmax_cross_entropy_with_logits_stable_1d, softmax_cross_entropy_with_logits_stable_1d')

  let softmax_cross_entropy_with_logits_2d =
    (softmax_cross_entropy_with_logits_stable_2d, softmax_cross_entropy_with_logits_stable_2d')

  let cross_entropy =
    (cross_entropy_1d, cross_entropy_1d')

}
