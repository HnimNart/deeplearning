import "network_types"

module type neural_network  = {

  type t
  type NN

  -- val add_loss_func: NN -> i32 -> NN
  -- val add_train_func: NN -> i32 -> NN
  -- val add_prediction_func: NN -> i32 -> NN

  val get_weights: NN -> []t
  val get_w_index: NN -> [](i32,i32)
  val get_b_index: NN -> [](i32,i32)
  val get_w_cnt: NN -> i32

  val get_loss_func: NN -> i32
  val get_predict_func: NN -> i32
  val get_train_func: NN -> i32
  val get_nn_depth: NN -> i32
  val get_nn_output: NN -> i32
  val get_nn_output_i: NN -> [](i32, i32)


  val get_types: NN -> layer_types
  val get_activations: NN -> layer_activation
  val get_bias: NN -> layer_bias
  val get_layers_info: NN -> layer_infos
  val get_layers_index: NN -> layer_index
  val get_layers_length: NN -> []i32


  val empty_network: []i32 -> NN
  val connect_layer: NN -> layer -> NN

}

module neural_network (R:real): neural_network with t = R.t with NN = NN R.t =  {

  type t = R.t
  type NN = NN t

  let get_weights (nn:NN) = nn.data.weights
  let get_w_index (nn:NN) = nn.data.w_i
  let get_b_index (nn:NN) = nn.data.b_i
  let get_w_cnt (nn:NN)   = nn.data.w_cnt

  let get_loss_func (nn:NN)    = nn.info.loss
  let get_predict_func (nn:NN) = nn.info.predict
  let get_train_func (nn:NN)   = nn.info.train
  let get_nn_depth (nn:NN)     = nn.info.depth
  let get_nn_output (nn:NN)    = nn.info.output
  let get_nn_output_i (nn:NN)  = nn.info.output_i

  let get_types (nn:NN)        = nn.layers.tp
  let get_activations (nn:NN)  = nn.layers.act
  let get_bias (nn:NN)         = nn.layers.bias
  let get_layers_info (nn:NN)  = nn.layers.info
  let get_layers_index (nn:NN) = nn.layers.index
  let get_layers_length (nn:NN) = nn.layers.info_length


  let empty_network (dims:[]i32): NN =
    let data: nn_data t      = {weights= [], w_i = [(0,0)], b_i = [(0,0)], w_cnt = 0}
     let layer_info:nn_layers = {tp=[0], act=[0], bias=[false], info=dims , index=[0], info_length = [length dims]}
     let nn_info: nn_info     = {loss = 0, predict = 0, train = 0, depth = 0, output = reduce (*) 1 dims, output_i = [(0, reduce (*) 1 dims)]}
     in {data=data, info = nn_info, layers=layer_info}

  let connect_layer (nn:NN) (layer:layer) : NN =
    let len_info    = length layer.info
    let len_info_i  = length nn.layers.index - 1
    let prev_l_len  = nn.layers.info_length[len_info_i]
    -- Layers info
    let tp'         = concat nn.layers.tp [layer.tp]
    let act'        = concat nn.layers.act [layer.activation]
    let info'       = concat nn.layers.info layer.info
    let use_bias'   = concat nn.layers.bias [layer.use_bias]

    let index'      = concat nn.layers.index [nn.layers.index[len_info_i] + prev_l_len]
    let length'     = concat nn.layers.info_length [len_info]
    let layers'   = {tp = tp', act = act' , bias = use_bias', info = info', index = index', info_length = length'}
    -- NN info
    let depth'   = nn.info.depth + 1
    let output'  = nn.info.output + layer.info[1]  --- Should be adjusted to each layer type
    let layer_output  = layer.info[1]
    let (_,prev_l_output_i) = nn.info.output_i[len_info_i]
    let output_i'     = concat nn.info.output_i [(prev_l_output_i, prev_l_output_i + layer_output)]
    let nn_info' = {loss= nn.info.loss, predict = nn.info.predict, train = nn.info.train,
                    depth = depth', output = output', output_i = output_i'}

    --- NN data
    let (_, i)        = nn.data.b_i[nn.info.depth]
    let w_cnt'        = nn.data.w_cnt + layer.info[1] * layer.info[0] + (if layer.use_bias then layer.info[1] else 0)
    let (w_i_start,w_i_end)   = (i, i + layer.info[1] * layer.info[0])
    let (b_i_start,b_i_end)   = if layer.use_bias then (w_i_end, w_i_end + layer.info[1]) else (w_i_end, w_i_end)
    let w_i'  = concat nn.data.w_i [(w_i_start, w_i_end)]
    let b_i'  = concat nn.data.b_i [(b_i_start, b_i_end)]
    let data'  = {weights = nn.data.weights, w_i = w_i', b_i = b_i', w_cnt = w_cnt'}

   in {data = data', info = nn_info', layers = layers'}

}
