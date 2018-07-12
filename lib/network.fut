import "nn_types"
import "layers/layers"
import "activations"



module type network = {

  type t
  type ^updater

  val identity : ([]t -> []t, []t -> []t)
  val sigmoid  : ([]t -> []t, []t -> []t)
  val relu     : ([]t -> []t, []t -> []t)
  val tanh     : ([]t -> []t, []t -> []t)
  val softmax  : ([]t -> []t, []t -> []t)


  val connect_layers 'w1 'w2 'i1 'o1 'o2 'g1 'g2 'e1 'e2 'e22 : NN i1 w1 o1 g1 e22 e1 updater -> NN o1 w2 o2 g2 e2 e22 updater -> NN i1 (w1, w2) (o2) (g1,g2) (e2) (e1) updater
  val predict 'w 'g 'i 'e1 'e2 '^u 'o  : NN ([]i) (w) ([]o) g e1 e2 u -> []i -> (o -> o, o -> o)   ->  []o
  val accuracy 'w 'g 'e1 'e2 'i '^u 'o  :NN ([]i)  w  ([]o) g e1 e2 u ->   []i -> []o -> (o -> o, o -> o) -> (o -> i32) -> t
  val loss 'w 'g 'e1 'e2 '^u 'i 'o : NN ([]i) w ([]o) g e1 e2 u -> []i -> []o ->
                                      (o -> o -> t, o -> o -> o) -> (o -> o, o -> o) -> t
  val argmax : []t -> i32
  val argmin : []t -> i32
}


module neural_network (R:real): network  with t = R.t
                                         with updater = updater ([][]R.t, []R.t)
                                         = {

  type t = R.t
  type updater = updater ([][]t, []t)
  type activation_func 'o = (o -> o, o -> o)
  type loss_func 'o = (o -> o -> t, o -> o -> o)

  module act_funcs = activation_funcs R

  let connect_layers 'w1 'w2 'i1 'o1 'o2 'g1 'g2 'e1 'e2 'e22  ((f1, b1, u1,ws1): NN i1 w1 o1 g1 e22 e1 updater)
                                                               ((f2, b2, u2,ws2): NN o1 w2 o2 g2 e2 e22 updater)
                                                               : NN i1 (w1,w2) (o2) (g1,g2) (e2) (e1) (updater) =

    ((\(training) (w1, w2) (input) ->
                            let (g1, res)  = f1 training w1 input
                            let (g2, res2) = f2 training w2 res
                            in ((g1, g2), res2)),
     (\(w1,w2) (g1,g2) (error) ->
                            let (err2, w2') = b2 w2 g2 error
                            let (err1, w1') = b1 w1 g1 err2
                            in (err1, (w1', w2'))),
     (\(f) (w1, w2) (wg1, wg2)  ->
                            let w1' = u1 f w1 wg1
                            let w2' = u2 f w2 wg2
                            in (w1', w2')),
     (ws1, ws2))


  let predict  'i 'w 'g 'e1 'e2 'u 'o  ((f,_, _ ,w):NN ([]i) w ([]o) g e1 e2 u) (input:[]i) (classification:activation_func o)  =
    let (_, output) = f false w input
    in map (\o -> classification.1 o) output

  let accuracy [d] 'w 'g 'e1 'e2 'u 'i 'o (nn:NN ([]i) w ([]o) g e1 e2 u) (input:[d]i) (labels:[d]o)
                                          (classification:activation_func o) (f: o -> i32)  : t =
    let predictions          = predict nn input classification
    let argmax_labels        = map (\x -> f x) labels
    let argmax_predictions   = map (\x -> f x) predictions
    let total                = reduce (+) 0 (map2 (\x y -> if x == y then 1 else 0)
                                             argmax_labels argmax_predictions)
    in R.(i32 total / i32 d)

  let loss [d] 'w 'g 'e1 'e2 'u 'i 'o (nn:NN ([]i) w ([]o) g e1 e2 u) (input:[d]i) (labels:[d]o)
                                      (loss: loss_func o) (classification:activation_func o) =
    let predictions = predict nn input classification
    let loss        = map2 (\p l -> loss.1 p l) predictions labels
    in reduce (R.+) R.(i32 0) loss

  let argmax [n] (X:[n]t) : i32 =
    reduce (\n i -> if unsafe R.(X[n] > X[i]) then n else i) 0 (iota n)

  let argmin [n] (X:[n]t) : i32 =
    reduce (\n i -> if unsafe R.(X[n] > X[i]) then n else i) 0 (iota n)


  let identity = act_funcs.Identity_1d
  let sigmoid  = act_funcs.Sigmoid_1d
  let relu     = act_funcs.Relu_1d
  let tanh     = act_funcs.Tanh_1d
  let softmax  = act_funcs.Softmax_1d

  -- let get_f (nn:layer) = nn.1
  -- let get_b (nn:layer) = nn.2
  -- let get_ws (nn:layer): weights = nn.4

}
