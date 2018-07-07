

module type layer = {

  type t
  type input
  type weights
  type output
  type error_in
  type error_out
  type gradients

  type act
  type layer
  type input_params

  -- val forward: act -> weights -> input -> output
  -- val backward:  act -> bool ->  weights ->  input -> error_in -> gradients
  val layer: input_params -> (act, act) -> layer

  -- val get_ws: layer -> weights
  -- val get_f: layer -> weights -> input -> (input, output)
  -- val get_b: layer -> bool -> weights ->  input -> error_in -> gradients

}
