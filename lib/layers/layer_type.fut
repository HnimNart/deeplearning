

module type layer = {

  type t
  type input
  type weights
  type output
  type error_in
  type error_out
  type gradients

  type ^act
  type ^layer
  type input_params

  val init: input_params -> (act, act) -> i32 -> layer

}
