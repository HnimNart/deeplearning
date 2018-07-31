import "../nn_types"
import "optimizer_type"
import "gradient_descent"


-- | Collection module for optimizers to be accessed through
module type optimizers =  {

  type t
  val gradient_descent 'i 'w 'g 'o 'e2 : NN ([]i) w ([]o) g ([]o) (e2) (apply_grad t)
                                       -> t
                                       -> ([]i)
                                       -> ([]o)
                                       -> i32
                                       -> loss_func o t
                                       -> NN ([]i) w ([]o) g ([]o) (e2) (apply_grad t)
}

module optimizers_coll (R:real) : optimizers with t = R.t = {


  type t = R.t
  module gd = gradient_descent R

  let gradient_descent 'w 'g  'i 'o  'e2 (nn:NN ([]i) w ([]o) g ([]o) (e2) (apply_grad t)) (alpha:t)
                      (input:[]i) (labels:[]o) (step_sz: i32) (loss:loss_func o t)  =
    gd.train nn alpha input labels step_sz loss

}
