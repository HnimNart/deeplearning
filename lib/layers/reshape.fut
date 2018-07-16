import "../nn_types"
import "layer_type"
import "/futlib/linalg"
import "../util"


module flatten (R:real) : layer with t = R.t
                                with input_params = ()
                                with activations  = ()
                                with layer        = flatten_tp R.t = {

  type t = R.t
  type input        = arr4d t
  type weights      = ()
  type output       = arr2d t
  type garbage      = (i32, i32, i32)
  type error_in     = arr2d t
  type error_out    = arr4d t
  type gradients    = (error_out, weights)
  type input_params = ()
  type activations  = ()

  type layer = flatten_tp t

  let empty_garbage: garbage = (0, 0, 0)

  let forward (training: bool) (_:weights) (input:input) : (garbage, output) =
     let dims = (length input[0], length input[0,0], length input[0,0,0])
     let garbage = if training then dims else empty_garbage
     in (garbage, map (\image -> flatten_3d image) input)

  let backward (_: weights) (input:garbage) (error:error_in) : gradients =
    let (p, m,n) =  input
    let retval = map (\img -> unflatten_3d p m n img) error
    in (retval, ())

  let update (_:apply_grad t) (_:weights) (_:weights)  = ()

  let init () () (_:i32) =
    (forward,
     backward,
     update,
     ())

}
