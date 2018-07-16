module type layer = {

  type t

  type ^layer
  type input_params
  type ^activations

  --- Initializa layer given input params
  val init: input_params -> activations -> i32 -> layer

}
