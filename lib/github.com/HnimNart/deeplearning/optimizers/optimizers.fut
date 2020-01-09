import "../nn_types"
import "optimizer_type"
import "gradient_descent"


-- | Collection module for optimizers to be accessed through
module type optimizers =  {

  type t
  val gradient_descent [n][m][K] 'i 'w 'g 'o 'e2 :
    NN ([n]i) w ([m]o) g ([m]o) e2 (apply_grad3 t)
    -> t
    -> ([K][n]i)
    -> ([K][m]o)
    -> i32
    -> loss_func ([m]o) t
    -> NN ([n]i) w ([m]o) g ([m]o) e2 (apply_grad3 t)
}

module optimizers_coll (R:real) : optimizers with t = R.t = {


  type t = R.t
  module gd = gradient_descent R

  let gradient_descent [n][m][K] 'w 'g 'o 'e2 'i
                      (nn: NN ([n]i) w ([m]o) g ([m]o) e2 (apply_grad3 t)) (alpha:t)
                      (input: [K][n]i) (labels: [K][m]o) (step_sz: i32)
                      (loss: loss_func ([m]o) t) =
    gd.train nn alpha input labels step_sz loss
}
