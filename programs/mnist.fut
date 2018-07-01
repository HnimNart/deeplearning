import "../lib/tensorflow"

module tf = tensorflow f32

let layer1 = tf.layers.fully_connected_2d [784, 256]
let layer2 = tf.layers.fully_connected_2d [256, 128]
let output = tf.layers.fully_connected_2d [128, 10]

let nn0   = tf.nn.empty_network()
let nn1   = tf.nn.connect_layer nn0 layer1
let nn2   = tf.nn.connect_layer nn1 layer2
let nn3   = tf.nn.connect_layer nn2 output

let model = tf.nn.init_network_w_rand_norm nn3 2

let main [m][n][d] (input: [m][d]tf.t) (labels: [m][n]tf.t) =
  let data_sets = 64000
  let batch_size = 128
  let i = 0
  let (nn,_ ) = loop (model, i) while i < data_sets do
                    let model: tf.nn.NN =  tf.optimizer.train_batch model input[i:i+batch_size] labels[i:i+batch_size] 0.1
                    in (model, i + batch_size)
                     -- in (tf.nn.get_bias nn)
                    in(tf.nn.accuracy nn input[:data_sets] labels[:data_sets])
                        -- tf.nn.accuracy nn input[:data_sets] labels[:data_sets],
                        -- tf.nn.accuracy model input[:data_sets] labels[:data_sets])



-- in concat next_layer_calc nn input_from_cur_layer i+1