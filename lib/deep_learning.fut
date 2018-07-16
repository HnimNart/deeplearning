import "nn_types"
import "neural_network"
import "activations_funcs"
import "trainers/optimizers"
import "layers/layers"
import "loss_funcs"

module deep_learning (R:real) : {


  type t = R.t
  module nn     : network    with t = R.t
  module layers : layers     with t = R.t
  module train  : optimizers with t = R.t

  module activation : activations with t = R.t
                                  with act_pair_1d = f_pair_1d R.t
                                  with act_pair_2d = f_pair_2d R.t

  module loss: loss with t = R.t
                    with loss_1d      = (arr1d R.t -> arr1d R.t -> R.t)
                    with loss_2d      = (arr2d R.t -> arr2d R.t -> R.t)
                    with loss_diff_1d = (arr1d R.t -> arr1d R.t -> arr1d R.t)
                    with loss_diff_2d = (arr2d R.t -> arr2d R.t -> arr2d R.t)

} = {

  type t = R.t
  module nn = neural_network R
  module layers = layers_coll R
  module activation = activation_funcs R
  module loss = loss_funcs R
  module train = optimizers_coll R

}