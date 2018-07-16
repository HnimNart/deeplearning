import "../nn_types"
import "optimizer_types"
import "GradientDescent"

module type optimizers =  {

  type t

  val GradientDescent 'i 'w 'g 'o 'e2 : NN ([]i) w ([]o) g ([]o) (e2) (apply_grad t) -> t -> ([]i) -> ([]o) ->
                                       i32 -> (o -> o -> t, o -> o -> o)
                                       ->  NN ([]i) w ([]o) g ([]o) (e2) (apply_grad t)

}


module optimizers_coll (R:real) : optimizers with t = R.t = {


  type t = R.t
  module gsd = GradientDescent R


  let GradientDescent 'w 'g  'i 'o  'e2 (nn:NN ([]i) w ([]o) g ([]o) (e2) (apply_grad t)) (alpha:t)
                      (input:[]i) (labels:[]o) (step_sz: i32) (loss:(o -> o -> t , o -> o -> o))  =
    gsd.train nn alpha input labels step_sz loss

}
