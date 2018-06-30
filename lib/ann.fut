import "layers"
import "util"
import "optimizer"
import "classifications"
import "loss"

module type network = {

  type t
  type NN

  val empty_network : () -> NN
  val init_network_w_rand_norm: NN-> i32 -> NN
  val connect_layer: NN -> []i32 -> NN
  -- val add_loss_func: NN_tp -> i32 -> NN_tp
  -- val add_classification_func: NN_tp -> i32 -> NN_tp

  val get_weights: NN -> []t
  val get_bias: NN -> []t
  val get_dims: NN -> [](i32, i32)
  val get_types: NN -> []i32
  val get_index: NN -> [](i32, i32)
  val get_outputs: NN -> [](i32,i32)


  val loss_batch: NN -> [][]t -> [][]t -> t
  val predict: NN -> [][]t -> i32 -> [][]t
  val accuracy:NN ->  [][]t -> [][]t -> t

}

module neural_network (R:real): network with t = R.t with NN = NN R.t = {

  type t = R.t
  type NN = NN t

  module random = normal_random_array R
  module optimizer = gradient_descent R
  module class_funcs = class_funcs_coll R
  module loss_funcs = loss_funcs R



   --- Returns an empty network
  let empty_network () : NN =
    let w'     : []t = []
    let b'     : []t = []
    let index' : [](i32,i32) = []
    let dims'  : [](i32, i32) = []
    let types' : []i32 = []
    let nn_output : [](i32, i32) = []
    let info' = {tp = types', dims = dims', index=  index'}
    in {data = {w = w', b = b'},  info = info', nn_outputs = nn_output}

  --- Connect layer to network
    let connect_layer (nn:NN) (info: []i32) =
    let dims'      = concat nn.info.dims [(info[1], info[2])]
    let tp'        = concat nn.info.tp [info[0]]
    let (_, i) =
      if length nn.info.index == 0
      then
        (0,0)
      else
        nn.info.index[(length nn.info.index) - 1]
    let index' = concat nn.info.index [(i, i + (info[1]*info[2]))]
    let info'  = {tp = tp', dims = dims', index = index'}
    in {data = nn.data, info = info', nn_outputs = nn.nn_outputs}

  ---- Initilize a network
  let init_network_w_rand_norm (nn: NN) (seed:i32): NN =
    let weights      = random.gen_random_array (nn.info.index[length nn.info.index - 1]).2 seed
    let (rows, cols) = unzip nn.info.dims
    let max_output'  = scan (\(x,y) (_,n) -> (x + (y-x), y + n)) (0,0) nn.info.dims
    let bias      = random.gen_random_array max_output'[(length max_output') - 1].2  seed
    in {data = {w = weights, b = bias}, info = nn.info, nn_outputs = max_output'}

  let get_weights (nn:NN) = nn.data.w
  let get_bias (nn:NN)  = nn.data.b
  let get_dims (nn:NN) = nn.info.dims
  let get_types (nn:NN) = nn.info.tp
  let get_index (nn:NN) = nn.info.index
  let get_outputs (nn:NN) = nn.nn_outputs

  let predict [m][n]  (nn:NN) (input:[m][n]t) (classes:i32) : [][]t =
    let output = optimizer.feed_forward nn input
    let output_len = length output[0]
    in map (\x -> class_funcs.calc_classification x[output_len-classes:] 1) output

  let is_equal [n] (logits:[n]t) (labels:[n]t) =
    let logits_i:i32 = (reduce (\n i -> if unsafe (R.(logits[n] > logits[i])) then n else i) 0 (iota n))
    let labels_i:i32 = (reduce (\n i -> if unsafe (R.(labels[n] > labels[i])) then n else i) 0 (iota n))
    in if logits_i == labels_i  then 1 else 0

  let accuracy [m][n][d] (nn:NN) (logits:[d][n]t) (labels:[d][m]t) =
    let predictions = predict nn logits 10
    let hits = map2 (\x y -> is_equal x y) predictions labels
    let total = reduce (+) 0 hits
    in R.(i32 total / i32 d)


  let loss_batch (nn: NN) (input: [][]t) (labels:[][]t) : t =
    let output = predict nn input 10
    let losses = map2 (\x y -> loss_funcs.calc_loss x y 1) labels output
    in reduce (R.+) R.(i32 0) losses


}
