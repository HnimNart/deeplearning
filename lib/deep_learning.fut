import "nn_types"
import "neural_network"
import "optimizers/optimizers"
import "layers/layers"
import "loss_funcs"

-- | Aggregation module for deep learning
module deep_learning (R:real) : {

  type t = R.t
  module nn     : network    with t = R.t
  module layers : layers     with t = R.t
  module train  : optimizers with t = R.t
  module loss: loss with t = R.t

} = {

  type t = R.t
  module nn = neural_network R
  module layers = layers_coll R
  module loss = loss_funcs R
  module train = optimizers_coll R
}
