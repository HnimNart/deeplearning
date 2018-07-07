import "../types"
import "layer_type"
import "dense"
import "reshape"


module layers (R:real) :{

  type t = R.t
  type dense_tp = NN ([][]t) ([][]t, [][]t) ([][]t) ([][]t) ([][]t) ([][]t) t
  module dense : layer with t = R.t
                       with input = [][]R.t
                       with weights = ([][]R.t, [][]R.t)
                       with output = [][]R.t
                       with error_in = ([][]R.t)
                       with error_out = ([][]R.t)
                       with gradients = ([][]R.t, ([][]R.t, [][]R.t))
                       with act = ([]R.t -> []R.t)
                       with layer = dense_tp

   val Dense: (i32, i32) -> (dense.act, dense.act) -> dense.layer

} = {

  type t = R.t
  type dense_tp = NN ([][]t) ([][]t, [][]t) ([][]t) ([][]t) ([][]t) ([][]t) t
  module dense = dense R


  let Dense ((m,n):(i32,i32)) (act_id: (dense.act, dense.act))  =
      dense.layer (m,n) act_id

  -- type act_pair_1d = i32
  -- type dense = NN

}
