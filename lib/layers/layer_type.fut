import "../nn_types"


module type layer = {

  type t

  type input_params
  type ^activations

  type input
  type output
  type weights
  type error_in
  type error_out
  type cache

  --- Initialize layer given input params
  val init: input_params -> activations -> i32 -> NN input weights output cache error_in error_out (apply_grad t)

}
