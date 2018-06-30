import "/futlib/linalg"

--- Single layer types
type layer_type   = i32
type layer_info   = []i32             -- info of a layer
type layer = {tp:i32, info:layer_info, activation: i32, use_bias: bool}


type data 't = {w:[]t}
type layer_types  = []layer_type      -- What the layer type is
type w_indexs = [](i32, i32)          -- Where each layers starts and ends in w
type b_indexs = [](i32, i32)          -- Where each bias starts in w
-- type layer_info = {tp:layer_types, dims:layer_dims, index: layer_indexs}


-- type NN 't = {data: data t, info: layer_info, nn_outputs: [](i32,i32)}

module type layer = {

  type t_arr
  type t

  val forwards: t_arr -> t_arr -> []t
  val backwards: []t -> []t

}


module dense (R:real) : layer with t = R.t with t_arr = ((i32, i32), []R.t) = {

  type t = R.t
  type t_arr = ((i32, i32), []R.t)
  module lalg = linalg R

  let forwards ((dim_x, x):t_arr) ((dim_w, w) :t_arr) =
    let X = unflatten dim_x.1 dim_x.2 x
    let W = unflatten dim_w.2 dim_w.1 w
    in flatten (lalg.matmul W X)

  let backwards (X:[]t) = X

}

-- module softmax (R:real) : layer with t = R.t = {

--   type t = R.t
--   let forwards [n] (_:[n]t) = map (\_ -> R.(i32 0)) (0..<n)
--   let backwards (X:[]t) = X
-- }
