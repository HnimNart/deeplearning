--- Single layer types
type layer_type   = i32
type layer_info   = []i32             -- info of a layer
type layer = {tp:i32, info:layer_info, activation: i32, use_bias: bool}

---- NN types
type weights 't = []t
type w_indexs = [](i32, i32)          -- Where each layers starts and ends in w
type b_indexs = [](i32, i32)          -- Where each bias starts in w

type layer_types      = []layer_type      -- What the layer type is
type layer_activation = []i32             -- What activation function a layer is using
type layer_bias       = []bool            -- whether a layer uses bias
type layer_infos      = []i32             -- Information all layers, i.e. dims etc
type layer_index      = []i32             -- What index a layer is in layer_infos

type nn_layers  = {tp:layer_types, act:layer_activation, bias:layer_bias, info:layer_infos, index:layer_index, info_length: []i32}
type nn_info    = {loss: i32, predict: i32, train: i32, depth :i32, output: i32, output_i: [](i32, i32)}
type nn_data 't = {weights:weights t, w_i: w_indexs, b_i:b_indexs, w_cnt: i32}

type NN 't = {data:nn_data t, info:nn_info, layers:nn_layers}
