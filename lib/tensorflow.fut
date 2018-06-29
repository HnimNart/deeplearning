import "ann"
import "layers"
import "optimizer"
-- import "ann_types"

module tensorflow (R:real): {

  type t = R.t
  module nn : network with t = R.t
  module layers : layers with t = R.t with input = ((i32, i32), []t)
  module optimizer : optimizer with t = R.t with NN = nn.NN
} = {

  -- Types
  type t = R.t
  --- Modules
  module nn = neural_network R
  module layers = layer R
  module optimizer = gradient_descent R
}
