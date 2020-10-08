import "../nn_types"
import "layer_type"
import "dense"
import "conv2d"
import "flatten"
import "max_pooling"

module type layers = {

  type t

  --- Layer types
  type^ dense_tp [m] [n] =
    dense_layer [m] [n] t

  type^ conv2d_tp [p][m][n] [filter_d] [filters] [out_m] [out_n] =
    conv2d_layer [p][m][n] [filter_d] [filters] [out_m] [out_n] t

  type^ max_pooling_tp [nlayer] [input_m][input_n] [output_m][output_n] =
    max_pooling_2d_layer [nlayer] [input_m][input_n] [output_m][output_n] t

  type^ flatten_tp [m][a][b] [n] =
    flatten_layer [m][a][b] [n] t

  -- Simple wrappers for each layer type
  val dense: (m: i64) -> (n: i64) -> activation_func ([n]t) ->  i32 -> dense_tp [m] [n]
  val conv2d : (p: i64) -> (m: i64) -> (n: i64)
            -> (filter_d: i64) -> (stride: i32) -> (filters: i64)
            -> (out_m: i64) -> (out_n: i64)
            -> ((d: i64) -> activation_func ([d]t))
            -> i32
            -> conv2d_layer [p][m][n] [filter_d] [filters] [out_m] [out_n] t
  val max_pooling2d :
             (nlayer: i64)
          -> (input_m: i64) -> (input_n: i64)
          -> (output_m: i64) -> (output_n: i64)
          -> max_pooling_tp [nlayer] [input_m][input_n] [output_m][output_n]
  val flatten : (m: i64) -> (a: i64) -> (b: i64) -> (n: i64)
          -> flatten_tp [m][a][b] [n]
}

module layers_coll (R:real): layers with t = R.t = {

  type t = R.t


  module dense_layer   = dense R
  module conv2d_layer  = conv2d R
  module maxpool_layer = max_pooling_2d R
  module flatten_layer = flatten R

  --- Layer types
  type^ dense_tp [m] [n] =
    dense_layer [m] [n] t

  type^ conv2d_tp [p][m][n] [filter_d] [filters] [out_m] [out_n] =
    conv2d_layer [p][m][n] [filter_d] [filters] [out_m] [out_n] t

  type^ max_pooling_tp [nlayer] [input_m][input_n] [output_m][output_n] =
    max_pooling_2d_layer [nlayer] [input_m][input_n] [output_m][output_n] t

  type^ flatten_tp [m][a][b] [n] =
    flatten_layer [m][a][b] [n] t

  let dense = dense_layer.init

  let conv2d = conv2d_layer.init

  let flatten = flatten_layer.init

  let max_pooling2d = maxpool_layer.init

}
