import "network_types"

module layers (R:real) : {

  type t = R.t
  type NN

  val dense: i32 -> i32 -> bool -> layer
  val conv2D: i32 -> (i32,i32) -> i32 -> i32 -> bool -> layer

  val empty_network: NN

} = {

  let dense_id  = 1
  let conv2d_id = 2
  let max_pooling2d_id = 3

  type t = R.t
  type NN = NN t

  let dense (neurons:i32) (activation:i32) (use_bias:bool):layer =
    {tp=dense_id, info=[neurons], activation=activation, use_bias=use_bias}

  let conv2D (filters:i32)
             ((kernel_m, kernel_n):(i32, i32))
             (padding: i32)
             (activation:i32)
             (use_bias:bool):layer =
    {tp=conv2d_id, info=[filters, kernel_m, kernel_n, padding],
     activation=activation, use_bias= use_bias}

  let empty_network : NN =
     let layer_info:nn_layers = {tp=[], act=[], bias=[], info=[], index=[]}
     let data: nn_data t      = {weights= [], w_i = [], b_i = []}
     in {data=data, info=layer_info}

  let connect_layer (nn:NN) (layer:layer) : NN =
    nn

}
