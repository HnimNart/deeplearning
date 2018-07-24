import "../nn_types"
import "optimizer_types"
import "GradientDescent"

module type optimizers =  {

  type t
  type loss_func 'o = {f:o -> o -> t, fd:o -> o -> o}
  val GradientDescent 'i 'w 'g 'o 'e2 : NN ([]i) w ([]o) g ([]o) (e2) (apply_grad t)
                                       -> t
                                       -> ([]i)
                                       -> ([]o)
                                       -> i32
                                       -> loss_func o
                                       -> NN ([]i) w ([]o) g ([]o) (e2) (apply_grad t)
}


module optimizers_coll (R:real) : optimizers with t = R.t = {


  type t = R.t
  type loss_func 'o = {f:o -> o -> t, fd:o -> o -> o}
  module gsd = GradientDescent R

  let GradientDescent 'w 'g  'i 'o  'e2 (nn:NN ([]i) w ([]o) g ([]o) (e2) (apply_grad t)) (alpha:t)
                      (input:[]i) (labels:[]o) (step_sz: i32) (loss:loss_func o)  =
    gsd.train nn alpha input labels step_sz loss

}
