import "../nn_types"
import "optimizer_types"
import "GradientDescent"

module type optimizers =  {

  type t

  type ^updater
  val GradientDescent 'i 'w 'g 'o 'e2 : NN ([]i) w ([]o) g ([]o) (e2) updater -> t -> ([]i) -> ([]o) ->
                                       i32 -> (o -> o -> t, o -> o -> o)
                                       ->  NN ([]i) w ([]o) g ([]o) (e2) updater

}


module optimizers_coll (R:real) : optimizers with t = R.t
                                             with updater = updater ([][]R.t, []R.t) ={


  type t = R.t
  type updater = updater ([][]t, []t)
  module gsd = GradientDescent R


  let GradientDescent 'w 'g  'i 'o  'e2 (nn:NN ([]i) w ([]o) g ([]o) (e2) updater) (alpha:t)
                      (input:[]i) (labels:[]o) (step_sz: i32) (loss:(o -> o -> t , o -> o -> o))  =
    gsd.train nn alpha input labels step_sz loss

}
