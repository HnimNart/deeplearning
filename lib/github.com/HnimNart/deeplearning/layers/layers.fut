import "../nn_types"
import "layer_type"
import "dense"
--import "conv2d"
import "flatten"
import "max_pooling"

module type layers = {

  type t

  -- type conv2d_cache
  -- type conv2d_weights
  -- type max_pooling2d_cache
  -- type max_pooling2d_weights

  --- Layer types
  type dense_tp [m] [n] = dense_layer [m] [n] t
  -- type conv2d_tp  =
  --   NN (arr4d t) conv2d_weights (arr4d t)
  --      conv2d_cache (arr4d t) (arr4d t) (apply_grad t)
  type max_pooling_tp [nlayer] [input_m][input_n] [output_m][output_n] =
    max_pooling_2d_layer [nlayer] [input_m][input_n] [output_m][output_n] t
  type flatten_tp [m][a][b] [n] = flatten_layer [m][a][b] [n] t

  -- Simple wrappers for each layer type
  val dense: (m: i32) -> (n: i32) -> activation_func ([n]t) ->  i32 -> dense_tp [m] [n]
  -- val conv2d: (i32, i32, i32, i32) -> (activation_func ([]t)) -> i32 -> conv2d_tp
  val max_pooling2d :
             (nlayer: i32)
          -> (input_m: i32) -> (input_n: i32)
          -> (output_m: i32) -> (output_n: i32)
          -> max_pooling_tp [nlayer] [input_m][input_n] [output_m][output_n]
  val flatten : (m: i32) -> (a: i32) -> (b: i32) -> (n: i32)
          -> flatten_tp [m][a][b] [n]
}

module layers_coll (R:real): layers with t = R.t = {

  type t = R.t


  module dense_layer   = dense R
  -- module conv2d_layer  = conv2d R
  module maxpool_layer = max_pooling_2d R
  module flatten_layer = flatten R

  -- type conv2d_weights = conv2d_layer.weights
  -- type conv2d_cache = conv2d_layer.cache

  --- Layer types
  type dense_tp [m] [n] = dense_layer [m] [n] t
  -- type conv2d_tp =
  --   NN (arr4d t) conv2d_weights (arr4d t)
  --      conv2d_cache (arr4d t) (arr4d t) (apply_grad t)
  type max_pooling_tp [nlayer] [input_m][input_n] [output_m][output_n] =
    max_pooling_2d_layer [nlayer] [input_m][input_n] [output_m][output_n] t
  type flatten_tp [m][a][b] [n] = flatten_layer [m][a][b] [n] t

  let dense (m: i32) (n: i32)
            (act_id: activation_func ([n]t))
            (seed:i32)  =
    dense_layer.init m n act_id seed

  -- let conv2d (params:conv2d_layer.input_params)
  --            (act:conv2d_layer.activations)
  --            (seed:i32) =
  --   conv2d_layer.init params act seed

  let flatten = flatten_layer.init

  let max_pooling2d = maxpool_layer.init

}
