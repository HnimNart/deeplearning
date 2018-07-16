import "../nn_types"

--- Layer types
--- Dense
type dense_tp 't =
    NN (arr2d t) (arr2d t,arr1d t) (arr2d t) ((arr2d t, arr2d t)) (arr2d t) (arr2d t) (apply_grad t)
--- Conv2d
type conv2d_tp 't =
    NN (arr4d t) (arr2d t,arr1d t) (arr4d t) ((i32, i32, i32), arr3d  t,arr4d  t) (arr4d  t) (arr4d  t) (apply_grad t)
--- Max pooling
type max_pooling_tp 't = NN (arr4d t) () (arr4d t) (arr4d (i32)) (arr4d t) (arr4d t) (apply_grad t)
--- Flatten
type flatten_tp 't =  NN (arr4d t) () (arr2d t) (i32, i32, i32) (arr2d t) (arr4d t) (apply_grad t)


module type layer = {

  type t

  type ^layer
  type input_params
  type ^activations

  --- Initialize layer given input params
  val init: input_params -> activations -> i32 -> layer

}
