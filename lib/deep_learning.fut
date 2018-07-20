import "nn_types"
import "neural_network"
import "activations_funcs"
import "optimizers/optimizers"
import "layers/layers"
import "loss_funcs"


-- | Aggregation module for deep learning
module deep_learning (R:real) : {

  type t = R.t
  module nn     : network    with t = R.t
  module layers : layers     with t = R.t
  module train  : optimizers with t = R.t

  module activation : activations with t = R.t
                                  with act_pair_1d = f_pair_1d R.t

  module loss: loss with t = R.t
                    with loss_1d      = loss_pair_1d R.t
} = {

  type t = R.t
  module nn = neural_network R
  module layers = layers_coll R
  module activation = activation_funcs R
  module loss = loss_funcs R
  module train = optimizers_coll R
}