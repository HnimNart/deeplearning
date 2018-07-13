import "../nn_types"
import "layer_type"
import "dense"
import "conv2d"
import "reshape"
import "max_pooling"

module type layers = {

  type t
  type dense_tp
  type conv2d_tp
  type flatten_tp
  type max_pooling2d_tp

  type ^act_1d

  val Dense: (i32, i32) -> act_1d -> i32 -> dense_tp
  val Conv2d: (i32, i32, i32, i32) -> act_1d -> i32 -> conv2d_tp
  val Max_pooling2d: (i32, i32) -> max_pooling2d_tp
  val Flatten: flatten_tp

}


module layers_coll (R:real): layers
                             with t = R.t
                             with act_1d   = ([]R.t -> []R.t, []R.t -> []R.t)
                             with dense_tp = NN ([][]R.t) ([][]R.t, []R.t) ([][]R.t) ([][]R.t, [][]R.t) ([][]R.t) ([][]R.t) (updater ([][]R.t, []R.t))
                             with max_pooling2d_tp = NN ([][][][]R.t) () ([][][][]R.t) ([][][][](i32, i32)) ([][][][]R.t) ([][][][]R.t) (updater ([][]R.t, []R.t))
                             with flatten_tp  = NN ([][][][]R.t) () ([][]R.t) (i32, i32, i32, i32) ([][]R.t) ([][][][]R.t) (updater ([][]R.t, []R.t))
                             with conv2d_tp   = NN ([][][][]R.t) ([][]R.t,[]R.t) ([][][][]R.t) ((i32, i32, i32),[][][]R.t, [][][][]R.t) ([][][][]R.t) ([][][][]R.t) (updater ([][]R.t, []R.t))
                               = {

  type t = R.t
  module dense        = dense R
  module conv2d       = conv2d R
  module flatten      = flatten R
  module maxpooling2d = max_pooling_2d R

  type updater     = updater ([][]t, []t)
  type dense_tp    = NN ([][]t) ([][]t, []t) ([][]t) ([][]t,[][]t) ([][]t) ([][]t) updater
  type conv2d_tp   = NN ([][][][]R.t) ([][]R.t,[]R.t) ([][][][]R.t) ((i32, i32, i32), [][][]R.t, [][][][]R.t) ([][][][]R.t) ([][][][]R.t) updater
  type flatten_tp  = NN ([][][][]R.t) () ([][]R.t) (i32, i32, i32, i32) ([][]R.t) ([][][][]R.t) updater
  type max_pooling2d_tp = NN ([][][][]R.t) () ([][][][]R.t) ([][][][](i32, i32)) ([][][][]R.t) ([][][][]R.t) updater

  type act_1d   = ([]t -> []t, []t -> []t)

  let Dense ((m,n):(i32,i32)) (act_id: act_1d) (seed:i32)  =
      dense.init (m,n) act_id seed

  let Conv2d (params:(i32, i32, i32, i32)) (act:act_1d) (seed:i32) =
      conv2d.init params act seed

  let Flatten = flatten.init () ((),()) 0

  let Max_pooling2d (params:(i32, i32)) =
     maxpooling2d.init params ((), ()) 0


}
