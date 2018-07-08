import "../types"
import "layer_type"
import "../activations"
import "/futlib/linalg"
import "../util"



module flatten (R:real) : layer with t = R.t
                                with input = [][][][]R.t
                                with input_params = ()
                                with weights = ()
                                with output = [][]R.t
                                with error_in = ([][]R.t)
                                with error_out = ([][][][]R.t)
                                with gradients = ([][][][]R.t, ())
                                with layer = NN ([][][][]R.t) () ([][]R.t) ([][][][]R.t) ([][]R.t) ([][][][]R.t) R.t
                                with act = () = {


  type t = R.t
  type input = [][][][]t
  type weights  = ()
  type output = [][]t
  type garbage  = [][][][]t
  type error_in = [][]t
  type error_out = [][][][]t
  type gradients = (error_out, weights)
  type input_params = ()
  type act = ()

  type layer = NN input weights output garbage error_in error_out t


   let forward (_:weights) (input:input) : output =
     map (\image -> flatten_3d image) input

  let backward (_:bool) (_: weights) (input:input) (error:error_in) : gradients =
    let (m,n, p, q) = (length input[0,0], length input[0,0,0], length input[0], length input)
    let err_flat = intrinsics.flatten (transpose error)
    in (unflatten_4d q p m n err_flat, ())

  let update (_:t) (_:weights) (_:weights)  = ()

  let layer () ((),()) =
    (\w input -> (input, forward w input),
     (backward ),
     update,
      (()))

}
