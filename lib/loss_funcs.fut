import "activation_funcs"
import "nn_types"

-- | Loss functions are defined as a record
--   of two function i.e.
--   1. The function f and
--   2. it's derivative w.r.t. to logits, fd
module type loss = {
  type t

  val cross_entropy : loss_func ([]t) t
  val softmax_cross_entropy_with_logits :loss_func ([]t) t
  val sum_of_squares_error : loss_func ([]t) t
}

module loss_funcs (R:real) : loss with t = R.t = {

  type t = R.t

  module activations = activation_funcs R

  let epsilon:t = R.(i32 1/ i32 10000000)

  let cross_entropy_1d [d] (logits:[d]t) (labels:[d]t) =
    let res = map2 (\x y -> R.((log (x + epsilon)) * y)) logits labels
    in R.(negate (sum res))

  let cross_entropy_1d' [d] (logits:[d]t) (labels:[d]t) =
    map2 (\x y -> R.(negate y / x)) logits labels

  let softmax_cross_entropy_with_logits_stable_1d [d] (logits:[d]t) (labels:[d]t) =
    let softmax_res = activations.Softmax_1d.f logits
    in cross_entropy_1d softmax_res labels

  let softmax_cross_entropy_with_logits_stable_1d' [d] (logits:[d]t) (labels:[d]t) =
    let softmax_res = activations.Softmax_1d.f logits
    in map2 (\x y -> R.(y - x)) labels softmax_res

  let sum_of_squares_error_1d [d] (logits:[d]t) (labels:[d]t) =
    let res = R.sum (map2 (\x y -> R.((x-y)**(i32 2) )) logits labels)
    in R.(res / i32 2)

  let sum_of_squares_error_1d' [d] (logits:[d]t) (labels:[d]t) =
      map2 (\x y -> R.(x - y)) logits labels

  --- Wrappers for loss function pairs ---
  let sum_of_squares_error =
    {f = sum_of_squares_error_1d, fd = sum_of_squares_error_1d'}

  let softmax_cross_entropy_with_logits =
    {f = softmax_cross_entropy_with_logits_stable_1d, fd = softmax_cross_entropy_with_logits_stable_1d'}

  let cross_entropy =
    {f = cross_entropy_1d, fd = cross_entropy_1d'}

}
