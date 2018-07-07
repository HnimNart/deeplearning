import "types"
import "layers/layers"
import "activations"
import "classifications"


module network (R:real) : {

  type t = R.t

  val combine 'w1 'w2 'i1 'o1 'o2 'g1 'g2 'e1 'e2 'e22 : NN i1 w1 o1 g1 e22 e1 t -> NN o1 w2 o2 g2 e2 e22 t -> NN i1 (w1, w2) (o2) (g1,g2) (e2) (e1) t
  val predict 'w 'g 'i 'e1 'e2 : NN ([]i) (w) ([][]t) (g) e1 e2 t -> []i -> [][]t
  val accuracy 'w 'g 'e2 'i :NN ([]i) w ([][]t) g ([][]t) e2 t -> []i -> [][]t -> t

} = {

  type t = R.t


  module clas = classification R


  let combine 'w1 'w2 'i1 'o1 'o2 'g1 'g2 'e1 'e2 'e22  ((f1, b1, u1,ws1): NN i1 w1 o1 g1 e22 e1 t) ((f2, b2, u2,ws2):NN o1 w2 o2 g2 e2 e22 t)
                                                                                    : NN i1 (w1,w2) (o2) (g1,g2) (e2) (e1) (t) =
    ((\(w1, w2) (input) ->  let (g1, res)  = f1 w1 input
                            let (g2, res2) = f2 w2 res
                            in ((g1, g2), res2)),
     (\(last_l) (w1,w2) (g1,g2) (error) ->
                            let (err2, w2') = b2 last_l w2 g2 error
                            let (err1, w1') = b1 false w1 g1 err2
                            in (err1, (w1', w2'))),
     (\ (alpha) (w1, w2) (wg1, wg2)  ->
                            let w1' = u1 alpha w1 wg1
                            let w2' = u2 alpha w2 wg2
                            in (w1', w2')),
     (ws1, ws2))


  let predict  'i 'w 'g 'e1 'e2 'u ((f,_, _ ,w):NN ([]i) w ([][]t) g e1 e2 u) (input:[]i)  =
    let (_, output) = f w input
    in clas.softmax_2d output

  let is_equal [n] (logits:[n]t) (labels:[n]t) =
    let logits_i:i32 = (reduce (\n i -> if unsafe (R.(logits[n] > logits[i])) then n else i) 0 (iota n))
    let labels_i:i32 = (reduce (\n i -> if unsafe (R.(labels[n] > labels[i])) then n else i) 0 (iota n))
    in if logits_i == labels_i  then 1 else 0


  let accuracy [d][c] 'w 'g 'e2 'u 'i (nn:NN ([]i) w ([][]t) g ([][]t) e2 u) (input:[d]i) (labels:[d][c]t) : t=
    let predictions = (predict nn input)
    let labels_T    = (labels)
    let hits        = map2 (\x y -> is_equal x y) predictions labels_T
    let total       = reduce (+) 0 hits
    in R.(i32 total / i32 d)

}
