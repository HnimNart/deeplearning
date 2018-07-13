import "../nn_types"
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
                                with layer = NN ([][][][]R.t) () ([][]R.t) (i32, i32, i32, i32) ([][]R.t) ([][][][]R.t) (updater ([][]R.t, []R.t))
                                with act = () = {


  type t = R.t
  type input = [][][][]t
  type weights  = ()
  type output = [][]t
  type garbage  = (i32, i32, i32, i32)
  type error_in = [][]t
  type error_out = [][][][]t
  type gradients = (error_out, weights)
  type input_params = ()
  type act = ()

  type layer = NN input weights output garbage error_in error_out (updater ([][]t, []t))

  let empty_garbage: garbage = (0, 0, 0, 0)

  let forward (training: bool) (_:weights) (input:input) : (garbage, output) =
     let dims = (length input[0,0], length input[0,0,0], length input[0], length input)
     let garbage = if training then dims else empty_garbage
     in (garbage, map (\image -> flatten_3d image) input)

  let backward (_: weights) (input:garbage) (error:error_in) : gradients =
    let (m,n, p, _) =  input
    let retval = map (\x -> unflatten_3d p m n x) error
    in (retval, ())

  let update (_:updater ([][]R.t, []R.t)) (_:weights) (_:weights)  = ()

  let init () ((),()) (_:i32) =
    (forward,
     backward ,
     update,
     (()))

}
