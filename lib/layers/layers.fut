import "../nn_types"
import "layer_type"
import "dense"
import "conv2d"
import "reshape"
import "max_pooling"

module type layers = {

  type t

  val Dense: (i32, i32) -> (f_pair_1d t) ->  i32 -> dense_tp t
  val Conv2d: (i32, i32, i32, i32) -> (f_pair_1d t) -> i32 -> conv2d_tp t
  val Max_pooling2d: (i32, i32) -> max_pooling_tp t
  val Flatten: flatten_tp t

}

module layers_coll (R:real): layers with t = R.t = {

  type t = R.t
  module dense        = dense R
  module conv2d       = conv2d R
  module flatten      = flatten R
  module maxpooling2d = max_pooling_2d R

  let Dense ((m,n):(i32,i32)) (act_id: f_pair_1d t) (seed:i32)  =
    dense.init (m,n) act_id seed

  let Conv2d (params:(i32, i32, i32, i32)) (act:f_pair_1d t) (seed:i32) =
    conv2d.init params act seed

  let Flatten =
    flatten.init () () 0

  let Max_pooling2d (params:(i32, i32)) =
    maxpooling2d.init params () 0


}
