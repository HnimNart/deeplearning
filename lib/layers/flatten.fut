import "../nn_types"
import "layer_type"
import "/futlib/linalg"
import "../util"


module flatten (R:real) : layer with t = R.t
                                with input_params = ()
                                with activations  = ()
                                with input        = arr4d R.t
                                with weights      = ()
                                with output       = arr2d R.t
                                with cache        = dims3d
                                with error_in     = arr2d R.t
                                with error_out    = arr4d R.t = {

  type t = R.t
  type input        = arr4d t
  type weights      = ()
  type output       = arr2d t
  type cache        = dims3d
  type error_in     = arr2d t
  type error_out    = arr4d t

  type input_params = ()
  type activations  = ()

  type flatten = NN input weights output cache error_in error_out (apply_grad t)

  let empty_cache: cache = (0, 0, 0)
  let empty_error: error_out = [[[[]]]]

  let forward (training: bool) (_:weights) (input:input) : (cache, output) =
     let dims = (length input[0], length input[0,0], length input[0,0,0])
     let cache = if training then dims else empty_cache
     in (cache, map (\image -> flatten_3d image) input)

  let backward (first_layer:bool)
               (_: weights)
               (input:cache)
               (error:error_in) : (error_out, weights) =

    if first_layer then
      (empty_error, ())
    else
      let (p,m,n) = input
      let error' = map (\img -> unflatten_3d p m n img) error
      in (error', ())

  let update (_:apply_grad t) (_:weights) (_:weights)  = ()

  let init () () (_:i32) : flatten =
    {forward  = forward,
     backward = backward,
     update   = update,
     weights  = ()}
}
