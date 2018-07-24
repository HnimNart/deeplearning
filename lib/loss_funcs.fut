import "activations_funcs"
import "nn_types"

-- | Loss functions are defined as a record
--   of two function i.e.
--   1. The function itself and
--   2. it's derivative w.r.t. to logits
module type loss = {

  type t
  type ^loss_1d

  val cross_entropy : loss_1d
  val softmax_cross_entropy_with_logits : loss_1d
  val sum_of_squares_error : loss_1d
}

module loss_funcs (R:real) : loss with t = R.t
                                  with loss_1d = loss_func ([]R.t) R.t = {

  type t = R.t
  type loss_1d = loss_func ([]t) t

  module activations = activation_funcs R

  let cross_entropy_1d [d] (logits:[d]t) (labels:[d]t) =
    let res = map2 (\x y -> if R.(isinf (negate (log x)))
                            then R.(i32 0)
                            else R.((log x) * y)) logits labels
    in R.(negate (reduce (R.+) R.(i32 0) res))

  let cross_entropy_1d' [d] (logits:[d]t) (labels:[d]t) =
    map2 (\x y -> R.(negate y / x)  ) logits labels

  let softmax_cross_entropy_with_logits_stable_1d [d] (logits:[d]t) (labels:[d]t) =
    let softmax_res = activations.Softmax_1d.f logits
    in cross_entropy_1d softmax_res labels

  let softmax_cross_entropy_with_logits_stable_1d' [d] (logits:[d]t) (labels:[d]t) =
    let softmax_res = activations.Softmax_1d.f logits
    in map2 (\x y -> R.(y - x)) labels softmax_res

  let sum_of_squares_error_1d [d] (logits:[d]t) (labels:[d]t) =
    let res = reduce (R.+) R.(i32 0) (map2 (\x y -> R.((x-y)**(i32 2) )) logits labels)
    in R.(res / i32 2)

  let sum_of_squares_error_1d' [d] (logits:[d]t) (labels:[d]t) =
      map2 (\x y -> R.(x - y)) logits labels

  let sum_of_squares_error =
    {f = sum_of_squares_error_1d, fd = sum_of_squares_error_1d'}

  let softmax_cross_entropy_with_logits =
    {f = softmax_cross_entropy_with_logits_stable_1d, fd = softmax_cross_entropy_with_logits_stable_1d'}

  let cross_entropy =
    {f = cross_entropy_1d, fd = cross_entropy_1d'}

}
