import "util"
import "/futlib/linalg"
import "layers"
import "layer_types"
import "ann"
import "loss"

module type optimizer = {
  type t
  type NN

  val feed_forward: NN -> []t -> []t
  val feed_forward_batch: NN -> [][]t -> [][]t
  val train_batch:  NN -> [][]t -> [][]t -> [][]t

  val backprop_batch: NN -> [][]t -> [][]t -> [][]t
}

module gradient_descent (R:real): optimizer with t = R.t with NN = NN R.t = {

  type t = R.t
  type NN = NN t

  module layers = layer R
  module lalg   = linalg R
  module util   = utility_funcs R


  --- Returns output from each layer
  let feed_forward (nn:NN) (input: []t) =
    -- Indicies for iteration
    let retval_end = length input
    let retval_start = 0
    let nn_output = nn.nn_outputs[length nn.nn_outputs - 1].2 + retval_end
    let retval: *[]t =
      unsafe (map (\i -> if i < retval_end then input[i] else R.(i32 0)) (0..< nn_output))
    let (retval, _, _) = loop (retval, retval_start, retval_end)
      for i < length nn.info.tp do
        let (m,n)            = nn.info.dims[i]
        let (w_start, w_end) = nn.info.index[i]
        let (r_start, r_end) = nn.nn_outputs[i]
        let weights          = unflatten n m nn.data.w[w_start:w_end]
        let res              = unsafe copy (lalg.matmul weights (unflatten m 1 retval[retval_start:retval_end]))
        let retval_i_diff    = (r_end - r_start)
        let retval           = unsafe scatter retval (retval_end..<(retval_end + retval_i_diff)) (flatten res)
      in (retval, retval_start + m, retval_end + n)
        in retval

   let feed_forward_batch (nn:NN) (input:[][]t) =
     map (\x -> feed_forward nn x) input


  module class_funcs = loss_funcs R

   let backprop_batch (nn:NN) (input:[][]t) (labels:[][]t) =
     let l_iter = length nn.info.tp - 1
     let input_iter = length input[0]
     let w_i = length nn.data.w
     let b_i = length nn.data.b
     let (m,n) =  nn.info.dims[l_iter]
     let final_l_output = map (\x -> x[input_iter-n:input_iter]) input
     let final_l_error  =
       map2 (\x y -> class_funcs.calc_loss_deriv x y 1) labels final_l_output
     let final_l_delta = map (\x -> unflatten n 1 x ) final_l_error

     let (_, prev_l_n)   = nn.info.dims[l_iter-1]
     let input_iter      = input_iter - prev_l_n
     let prev_l_output
         = map (\x -> unflatten prev_l_n 1 x[input_iter-prev_l_n:input_iter]) input

     let final_l_grads   = map2 (\x y -> flatten (lalg.matmul x (transpose y)) ) final_l_delta prev_l_output
     let grads_w_i       = m * n
     let grads_b_i       = n
     let grads_w         = map (\x -> scatter (map (\_ -> R.(i32 0)) (0..<w_i)) (w_i-grads_w_i..<w_i) x) final_l_grads
     in grads_w







   let train_batch (nn:NN) (data:[][]t) (labels:[][]t) =
     let output = feed_forward_batch nn data
     in backprop_batch nn output labels
}
