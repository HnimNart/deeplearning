import "../nn_types"
import "optimizer_types"
import "GradientDescent"

module type optimizers =  {

  type t

  val GradientDescent 'i 'w 'g 'e2 : NN ([]i) w ([][]t) g ([][]t) e2 t -> t -> ([]i) -> ([][]t) -> i32 -> ([][]t -> [][]t -> [][]t)
                             ->  NN ([]i) w ([][]t) g ([][]t) e2 t



}



module optimizers_coll (R:real) : optimizers with t = R.t = {


  type t = R.t
  module gsd = GradientDescent R


  let GradientDescent 'w 'g 'e2 'i (nn:NN ([]i) w ([][]t) g ([][]t) e2 t) (alpha:t)
                      (input:[]i) (labels:[][]t) (step_sz: i32) (loss:[][]t -> [][]t -> [][]t) =

    gsd.train nn alpha input labels step_sz loss


}
