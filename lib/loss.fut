import "classifications"

module loss (R:real) : {

  type t = R.t
  val softmax_cross_entropy_with_logits: [][]t -> [][]t -> [][]t

} = {

  type t = R.t
  module classification = classification R

  let softmax_cross_entropy_with_logits_stable [d] (labels: [d][]R.t) (logits:[d][]R.t) =
    map2 (\xr yr -> map2 (\x y  -> R.(y - x)) xr yr) labels (classification.softmax_2d logits)

  let softmax_cross_entropy_with_logits =
     softmax_cross_entropy_with_logits_stable

}
