import "nn_types"
import "activations_funcs"

module type network = {

  type t

  --- activation function wrappers
  val identity : f_pair_1d t
  val sigmoid  : f_pair_1d t
  val relu     : f_pair_1d t
  val tanh     : f_pair_1d t
  val softmax  : f_pair_1d t

  --- helper functions for calculating accuracy
  val argmax : []t -> i32
  val argmin : []t -> i32

  --- Combines two 'networks' into one
  val connect_layers 'w1 'w2 'i1 'o1 'o2 'c1 'c2 'e1 'e2 'e22:
                      NN i1 w1 o1 c1 e22 e1 (apply_grad t) ->
                      NN o1 w2 o2 c2 e2 e22 (apply_grad t) ->
                      NN i1 (w1, w2) (o2) (c1,c2) (e2) (e1) (apply_grad t)
  --- Performs predictions on data set given a network, input data and classifier
  val predict 'w 'g 'i 'e1 'e2 '^u 'o  : NN ([]i) (w) ([]o) g e1 e2 u -> []i ->
                                         (o -> o, o -> o) -> []o
  --- Calculates the accuracy given a network, input, labels and classifier
  val accuracy 'w 'g 'e1 'e2 'i '^u 'o : NN ([]i)  w  ([]o) g e1 e2 u ->
                                         []i -> []o -> (o -> o, o -> o) -> (o -> i32) -> t
  --- Calculates the absolute loss given a network, input, labels, a loss function and classifier
  val loss 'w 'g 'e1 'e2 '^u 'i 'o : NN ([]i) w ([]o) g e1 e2 u -> []i -> []o ->
                                     (o -> o -> t, o -> o -> o) -> (o -> o, o -> o) -> t
}

module neural_network (R:real): network with t = R.t = {

  type t = R.t
  type activation_func 'o = (o -> o, o -> o)
  type loss_func 'o       = (o -> o -> t, o -> o -> o)

  module act_funcs = activation_funcs R

  let connect_layers 'w1 'w2 'i1 'o1 'o2 'c1 'c2 'e1 'e2 'e  ((f1, b1, u1, ws1): NN i1 w1 o1 c1 e e1 (apply_grad t))
                                                               ((f2, b2, u2, ws2): NN o1 w2 o2 c2 e2 e (apply_grad t))
                                                               : NN i1 (w1,w2) (o2) (c1,c2) (e2) (e1) (apply_grad t) =

    (\(training) (w1, w2) (input) ->
                            let (c1, res)  = f1 training w1 input
                            let (c2, res2) = f2 training w2 res
                            in ((c1, c2), res2),
     (\(_) (w1,w2) (c1,c2) (error) ->
                            let (err2, w2') = b2 false w2 c2 error
                            let (err1, w1') = b1 true w1 c1 err2
                            in (err1, (w1', w2'))),
     (\(f) (w1, w2) (wc1, wc2)  ->
                            let w1' = u1 f w1 wc1
                            let w2' = u2 f w2 wc2
                            in (w1', w2')),
     (ws1, ws2))

  let predict  'i 'w 'g 'e1 'e2 'u 'o  ((f,_, _ ,w):NN ([]i) w ([]o) g e1 e2 u) (input:[]i) (classifier:activation_func o)  =
    let (_, output) = f false w input
    in map (\o -> classifier.1 o) output

  let accuracy [d] 'w 'g 'e1 'e2 'u 'i 'o (nn:NN ([]i) w ([]o) g e1 e2 u) (input:[d]i) (labels:[d]o)
                                          (classification:activation_func o) (f: o -> i32) : t =
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

  --- Breaks if two or more values have max values?
  --- Question is which index should be chosen then?
  let argmax [n] (X:[n]t) : i32 =
    reduce (\n i -> if unsafe R.(X[n] > X[i]) then n else i) 0 (iota n)

  let argmin [n] (X:[n]t) : i32 =
    reduce (\n i -> if unsafe R.(X[n] > X[i]) then n else i) 0 (iota n)

  --- activation function wrappers
  let identity = act_funcs.Identity_1d
  let sigmoid  = act_funcs.Sigmoid_1d
  let relu     = act_funcs.Relu_1d
  let tanh     = act_funcs.Tanh_1d
  let softmax  = act_funcs.Softmax_1d

}
