import "nn"
import "layers"

-- import "ann_types"

module tensorflow (R:real): {

  type t = R.t
  module nn : neural_network with t = R.t
  module layers : layers with t = R.t with NN = nn.NN
  -- module optimizer : optimizer with t = R.t with NN = nn.NN
} = {

  -- Types
  type t = R.t
  --- Modules
  module nn = neural_network R
  module layers = layers R
  -- module optimizer = gradient_descent R
}



module tf = tensorflow f32

let input   = tf.nn.empty_network [784]
let layer_1 = tf.layers.dense [784,256] 0 true
let layer_2 = tf.layers.dense [256,256] 0 true
let out     = tf.layers.dense [256,10]  0 true

let nn0     = tf.nn.connect_layer input layer_1
let nn1     = tf.nn.connect_layer nn0 layer_2
let nn2     = tf.nn.connect_layer nn1 out

let main =
  let tmp = nn2
  let (w_i_m, w_i_n) =  unzip (tf.nn.get_w_index tmp)
  let (b_i_m, b_i_n) = unzip (tf.nn.get_b_index tmp)
  let (o_i_m, o_i_n) = unzip (tf.nn.get_nn_output_i tmp)
  in (tf.nn.get_weights tmp,
      w_i_m, w_i_n, b_i_m, b_i_n,
      tf.nn.get_w_cnt   tmp,
      tf.nn.get_loss_func tmp,
      tf.nn.get_predict_func tmp,
      tf.nn.get_train_func tmp,
      tf.nn.get_nn_depth tmp,
      tf.nn.get_nn_output tmp,
     o_i_m, o_i_n,



      tf.nn.get_types tmp ,
      tf.nn.get_bias tmp,
      tf.nn.get_activations tmp,
      tf.nn.get_layers_info tmp,
      tf.nn.get_layers_index tmp,
      tf.nn.get_layers_length tmp)