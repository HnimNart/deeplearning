import "types"
import "layers"
import "activations"
import "classifications"


module network (R:real) : {

  type t = R.t

  type dense_tp = NN ([][]t) ([][]t, [][]t) ([][]t) ([][]t) ([][]t) ([][]t)
  module dense : layer with t = R.t with input = [][]R.t with weights = ([][]R.t, [][]R.t) with output = [][]R.t
                       with error_in = ([][]R.t) with error_out = ([][]R.t) with gradients = ([][]R.t, ([][]R.t, [][]R.t))
                       with act = ([]R.t -> []R.t) with layer = dense_tp

  val combine 'w1 'w2 'i1 'o1 'o2 'g1 'g2 'e1 'e2 'e22: NN i1 w1 o1 g1 e22 e1 -> NN o1 w2 o2 g2 e2 e22 -> NN i1 (w1, w2) (o2) (g1,g2) (e2) (e1)


  val train 'w 'g 'e2: NN ([][]t) w ([][]t) g ([][]t) e2 -> [][]t -> [][]t -> t -> NN ([][]t) w ([][]t) g ([][]t) e2
  val accuracy 'w 'g 'e2 :NN ([][]t) w ([][]t) g ([][]t) e2 -> [][]t -> [][]t -> t
} = {

  type t = R.t

  type dense_tp = NN ([][]t) ([][]t, [][]t) ([][]t) ([][]t) ([][]t) ([][]t)

  module dense = dense R

  module cross = softmax_cross_entropy_with_logits R
  module soft = softmax_stable R

  let combine 'w1 'w2 'i1 'o1 'o2 'g1 'g2 'e1 'e2 'e22  ((f1, b1,ws1): NN i1 w1 o1 g1 e22 e1) ((f2, b2,ws2):NN o1 w2 o2 g2 e2 e22)
                                                                                    : NN i1 (w1,w2) (o2) (g1,g2) (e2) (e1) =
    ((\(w1, w2) (input) ->  let (g1, res)  = f1 w1 input
                            let (g2, res2) = f2 w2 res
                            in ((g1, g2), res2)),
     (\(w1,w2) (g1,g2) (error) ->
                            let (err2, w2') = b2  w2 g2 error
                            let (err1, w1') = b1 w1 g1 err2
                            in (err1, (w1', w2'))),
     (ws1, ws2))


  let getCols [m][n] (M:[m][n]t) (i:i32) (c:i32) : [m][c]t =
    M[:,i:i+c]

  let train 'w  'g 'e2  ((f,b,w):NN ([][]t) w ([][]t) g ([][]t) e2) (input:[][]t) (labels:[][]t) (alpha:t) =
    let i = 0
    let batch_size = 128
    let (w',_) = loop (w, i) while i < 64000 do
             let inp' = input[i:i+batch_size]-- getCols input i batch_size
             let lab  = labels[i:i+batch_size]-- getCols labels i batch_size
             let (os, output) = f w (inp')
             let error = cross.derivative (lab) output
             let (_, w') = b w os error
             in (w', i+ batch_size)
    in (f,b,w')

  let predict 'w 'g 'e2 ((f,_,w):NN ([][]t) w ([][]t) g ([][]t) e2) (input:[][]t)  =
    let (_, output) = f w input
    in soft.classify2 output

  let is_equal [n] (logits:[n]t) (labels:[n]t) =
    let logits_i:i32 = (reduce (\n i -> if unsafe (R.(logits[n] > logits[i])) then n else i) 0 (iota n))
    let labels_i:i32 = (reduce (\n i -> if unsafe (R.(labels[n] > labels[i])) then n else i) 0 (iota n))
    in if logits_i == labels_i  then 1 else 0

  let accuracy [d][c] 'w 'g 'e2 (nn:NN ([][]t) w ([][]t) g ([][]t) e2) (input:[d][]t) (labels:[d][c]t) : t=
    let predictions = (predict nn input)
    let labels_T    = (labels)
    let hits        = map2 (\x y -> is_equal x y) predictions labels_T
    let total       = reduce (+) 0 hits
    in R.(i32 total / i32 d)

}