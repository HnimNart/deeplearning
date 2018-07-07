import "loss"
import "types"
import "util"

module type optimizer = {

  type t

  val sgd 'i 'w 'g 'e2 : NN ([]i) w ([][]t) g ([][]t) e2 t -> t -> ([]i) -> ([][]t) -> i32
                             ->  NN ([]i) w ([][]t) g ([][]t) e2 t

}


module sgd (R:real) : optimizer with t = R.t = {

  type t = R.t

  module loss = loss R
  module util   = utility_funcs R


  let train 'w 'g 'e2 'i ((f,b,u,w):NN ([]i) w ([][]t) g ([][]t) e2 t) (alpha:t) (input:[]i) (labels:[][]t) (step_sz: i32)=
    let i = 0
    let (w',_) = loop (w, i) while i < length input - 1 do
             let inp' = input[i:i+step_sz]
             let lab  = labels[i:i+step_sz]
             let (os, output) = f w (inp')
             let error = loss.softmax_cross_entropy_with_logits lab output
             let (_, grads) = b true w os error
             let w'   = u alpha w grads
             in (w', i+ step_sz)
    in (f,b, u,w')

  let sgd 'w 'g 'e2 'i (nn:NN ([]i) w ([][]t) g ([][]t) e2 t) (alpha:t) (input:[]i) (labels:[][]t) =
     train (nn) (alpha) input labels

}
