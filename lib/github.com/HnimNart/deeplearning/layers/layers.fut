import "../nn_types"
import "layer_type"
import "dense"
import "conv2d"
import "flatten"
import "max_pooling"

module type layers = {

  type t

  type dense_cache
  type dense_weights
  type conv2d_cache
  type conv2d_weights
  type flatten_cache
  type flatten_weights
  type max_pooling2d_cache
  type max_pooling2d_weights

  --- Layer types
  type dense_tp =
    NN (arr2d t) dense_weights (arr2d t)
       dense_cache (arr2d t) (arr2d t) (apply_grad t)
  type conv2d_tp  =
    NN (arr4d t) conv2d_weights (arr4d t)
       conv2d_cache (arr4d t) (arr4d t) (apply_grad t)
  type max_pooling_tp  =
    NN (arr4d t) max_pooling2d_weights (arr4d t)
       max_pooling2d_cache (arr4d t) (arr4d t) (apply_grad t)
  type flatten_tp  =
    NN (arr4d t) flatten_weights (arr2d t) flatten_cache (arr2d t) (arr4d t) (apply_grad t)

  -- Simple wrappers for each layer type
  val dense: (i32, i32) -> (activation_func ([]t)) ->  i32 -> dense_tp
  val conv2d: (i32, i32, i32, i32) -> (activation_func ([]t)) -> i32 -> conv2d_tp
  val max_pooling2d: (i32, i32) -> max_pooling_tp
  val flatten: flatten_tp
}

module layers_coll (R:real): layers with t = R.t = {

  type t = R.t


  module dense_layer   = dense R
  module conv2d_layer  = conv2d R
  module flatten_layer = flatten R
  module maxpool_layer = max_pooling_2d R

  type dense_weights = dense_layer.weights
  type dense_cache = dense_layer.cache
  type conv2d_weights = conv2d_layer.weights
  type conv2d_cache = conv2d_layer.cache
  type flatten_weights = flatten_layer.weights
  type flatten_cache = flatten_layer.cache
  type max_pooling2d_weights = maxpool_layer.weights
  type max_pooling2d_cache = maxpool_layer.cache

  --- Layer types
  type dense_tp =
    NN (arr2d t) dense_weights (arr2d t)
       dense_cache (arr2d t) (arr2d t) (apply_grad t)
  type conv2d_tp =
    NN (arr4d t) conv2d_weights (arr4d t)
       conv2d_cache (arr4d t) (arr4d t) (apply_grad t)
  type max_pooling_tp =
    NN (arr4d t) max_pooling2d_weights (arr4d t)
       max_pooling2d_cache (arr4d t) (arr4d t) (apply_grad t)
  type flatten_tp =
    NN (arr4d t) flatten_weights (arr2d t)
       flatten_cache (arr2d t) (arr4d t) (apply_grad t)

  let dense ((m,n):dense_layer.input_params)
            (act_id:dense_layer.activations)
            (seed:i32)  =
    dense_layer.init (m,n) act_id seed

  let conv2d (params:conv2d_layer.input_params)
             (act:conv2d_layer.activations)
             (seed:i32) =
    conv2d_layer.init params act seed

  let flatten =
    flatten_layer.init () () 0

  let max_pooling2d (params:maxpool_layer.input_params) =
    maxpool_layer.init params () 0

}
