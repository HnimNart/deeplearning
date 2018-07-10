import "optimizer_types"
import "../nn_types"

module GradientDescent (R:real) : trainer with t = R.t = {

  type t = R.t

  let train 'w 'g 'e2 'i ((f,b,u,w):NN ([]i) w ([][]t) g ([][]t) e2 t) (alpha:t)
                          (input:[]i) (labels:[][]t) (step_sz: i32) (loss:[][]t -> [][]t -> [][]t) =

    let i = 0
    let (w',_) = loop (w, i) while i < length input - 1 do
             let inp' = input[i:i+step_sz]
             let lab  = labels[i:i+step_sz]
             let (os, output) = f w (inp')
             let error = loss output lab
             let (_, grads) = b w os (transpose error)
             let w'   = u alpha w grads
             in (w', i+ step_sz)
    in (f,b, u,w')

}