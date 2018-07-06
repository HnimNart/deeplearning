import "loss"
import "types"


module type optimizer = {

  type t

  val train  'w 'g 'e2 : NN ([][]t) w ([][]t) g ([][]t) e2 -> ([][]t) -> ([][]t) ->  NN ([][]t) w ([][]t) g ([][]t) e2

}


module sgd (R:real) : optimizer with t = R.t = {

  type t = R.t

  module loss = loss R

  let train 'w 'g 'e2  ((f,b,u,w):NN ([][]t) w ([][]t) g ([][]t) e2) (input:[][]t) (labels:[][]t) =
    let i = 0
    let batch_size = 128
    let (w',_) = loop (w, i) while i < 64000 do
             let inp' = input[i:i+batch_size]-- getCols input i batch_size
             let lab  = labels[i:i+batch_size]-- getCols labels i batch_size
             let (os, output) = f w (inp')
             let error = loss.softmax_cross_entropy_with_logits lab output
             let (_, grads) = b w os error
             let w'   = u w grads
             in (w', i+ batch_size)
    in (f,b, u,w')




}