import "nn_types"
import "network"
import "activations"
import "trainers/optimizers"
import "layers/layers"
import "loss"




module tensorflow (R:real) : {


  type t = R.t
  module nn : network with t = R.t
                      with updater = (updater ([][]t, []t))

  module layers: layers with t = R.t
                        with act_1d = ([]R.t -> []R.t, []R.t -> []R.t)
                        with dense_tp = NN ([][]R.t) ([][]R.t, []R.t) ([][]R.t) ([][]R.t, [][]R.t) ([][]R.t) ([][]R.t) (updater ([][]R.t, []R.t))
                        with max_pooling2d_tp = NN ([][][][]R.t) () ([][][][]R.t) ([][][][](i32, i32)) ([][][][]R.t) ([][][][]R.t)  (updater ([][]R.t, []R.t))
                        with flatten_tp  = NN ([][][][]R.t) () ([][]R.t) (i32, i32, i32, i32) ([][]R.t) ([][][][]R.t) (updater ([][]R.t, []R.t))
                        with conv2d_tp   = NN ([][][][]R.t) ([][]R.t,[]R.t) ([][][][]R.t) ((i32, i32, i32), [][][]R.t, [][][][]R.t) ([][][][]R.t) ([][][][]R.t) (updater ([][]R.t, []R.t))


  module train : optimizers with t = R.t
                            with updater = nn.updater


  module activation : activations with t = R.t
                                   with act_pair_1d = ([]R.t -> []R.t, []R.t -> []R.t)
                                   with act_pair_2d = ([][]R.t -> [][]R.t, [][]R.t -> [][]R.t)

  module loss: loss with t = R.t
                    with loss_1d      = ([]R.t -> []R.t -> R.t)
                    with loss_2d      = ([][]R.t -> [][]R.t -> R.t)
                    with loss_diff_1d = ([]R.t -> []R.t -> []R.t)
                    with loss_diff_2d = ([][]R.t -> [][]R.t -> [][]R.t)

} = {

  type t = R.t
  module nn = neural_network R
  module layers = layers_coll R
  module activation = activation_funcs R
  module loss = loss_funcs R
  module train = optimizers_coll R


}