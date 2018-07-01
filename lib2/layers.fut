import "network_types"


module type layer = {


  type t
  type NN
  val forward: NN -> [][]t -> i32 -> [][]t
  val backwards: NN -> [][]t -> i32 -> []t

}




module dense (R:real) : layer with t = R.t = {

  type t = R.t
  type NN = NN t

  let forward (nn:NN) (input:[][]t) (layer_i: i32) =
    input

  let backwards (nn:NN) (input:[][]t) (layer_i:i32) =
    input[0]

}

module type layers = {

  type t
  type NN

  val dense: []i32 -> i32 -> bool -> layer
  val conv2D: i32 -> (i32,i32) -> i32 -> i32 -> bool -> layer

}

module layers (R:real): layers with t = R.t with NN = NN R.t = {

  let dense_id  = 1
  let conv2d_id = 2
  let max_pooling2d_id = 3

  type t = R.t
  type NN = NN t

  let dense (dims:[]i32) (activation:i32) (use_bias:bool):layer =
    {tp=dense_id, info=dims, activation=activation, use_bias=use_bias}

  let conv2D (filters:i32)
             ((kernel_m, kernel_n):(i32, i32))
             (padding: i32)
             (activation:i32)
             (use_bias:bool):layer =
    {tp=conv2d_id, info=[filters, kernel_m, kernel_n, padding],
     activation=activation, use_bias= use_bias}



}
