import "../nn_types"
import "layer_type"
import "dense"
import "reshape"


module type layers = {

  type t
  type dense_tp -- = NN ([][]t) ([][]t, []t) ([][]t) ([][]t) ([][]t) ([][]t) t
  module dense : layer
                       -- with input = [][]R.t
                       -- with weights = ([][]R.t, []R.t)
                       -- with output = [][]R.t
                       -- with error_in = ([][]R.t)
                       -- with error_out = ([][]R.t)
                       -- with gradients = ([][]R.t, ([][]R.t, []R.t))
                       -- with act = ([]R.t -> []R.t)
                       -- with layer = dense_tp

  type ^act_1d
  val Dense: (i32, i32) -> act_1d -> i32 -> dense_tp

}


module layers_coll (R:real): layers with t = R.t
                                    with act_1d = ([]R.t -> []R.t, []R.t -> []R.t)
                                    with dense_tp = NN ([][]R.t) ([][]R.t, []R.t) ([][]R.t) ([][]R.t) ([][]R.t) ([][]R.t) R.t
                                    = {

  type t = R.t
  module dense = dense R
  type dense_tp = NN ([][]t) ([][]t, []t) ([][]t) ([][]t) ([][]t) ([][]t) t
  type act_1d = ([]t -> []t, []t -> []t)

  let Dense ((m,n):(i32,i32)) (act_id: act_1d) (seed:i32)  =
      dense.init (m,n) act_id seed


}
