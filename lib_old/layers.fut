type data 't = {w:[]t, b:[]t}
type layer_types  = []i32             -- What the layer type is
type layer_dims   = [](i32, i32)      -- how many weights a layer has
type layer_indexs = [](i32, i32)      -- Where each layers starts and ends
type layer_info = {tp:layer_types, dims:layer_dims, index: layer_indexs}
type NN 't = {data: data t, info: layer_info, nn_outputs: [](i32,i32)}

import "layer_types"

module type layers = {

  type t
  type input

  val fully_connected_2d : []i32 -> []i32
  val feed_forward_layer: i32 -> input -> input ->  []t

}

module layer (R:real) : layers with t = R.t with input = ((i32, i32) , []R.t) =  {

  let fully_conn_id = 1
  let softmax = 2

  module full_conn =  fully_conn R

  type t = R.t

  type input = ((i32, i32), []t)

  let fully_connected_2d (dims:[2]i32) =
    [fully_conn_id, dims[0], dims[1]]

  let feed_forward_layer (layer_id:i32) (input:input) (w:input) =
    if layer_id == 1 then full_conn.forwards input w
    else full_conn.forwards input w
}
