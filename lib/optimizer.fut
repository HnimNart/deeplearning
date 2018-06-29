import "util"
import "/futlib/linalg"
import "layers"
import "layer_types"
import "loss"
import "activations"

module type optimizer = {
  type t
  type NN

  val feed_forward: NN -> []t -> []t
  val feed_forward_batch: NN -> [][]t -> [][]t
  val train_batch:  NN -> [][]t -> [][]t -> t -> (NN, []t)
  val backprop_batch: NN -> [][]t -> [][]t -> t -> ([]t, []t)
}

module gradient_descent (R:real): optimizer with t = R.t with NN = NN R.t = {

  type t = R.t
  type NN = NN t

  module layers = layer R
  module lalg   = linalg R
  module util   = utility_funcs R
  module act_funcs   = activation_funcs_coll R
  module class_funcs = loss_funcs R

  --- Returns output from each layer
  let feed_forward (nn:NN) (input: []t) =
    -- Indicies for iteration
    let retval_end = length input
    let retval_start = 0
    let b_i = 0
    let nn_output = nn.nn_outputs[length nn.nn_outputs - 1].2 + retval_end
    let retval: *[]t =
      unsafe (map (\i -> if i < retval_end then input[i] else R.(i32 0)) (0..< nn_output))
    let (retval, _, _, _) = loop (retval, retval_start, retval_end, b_i)
      for i < length nn.info.tp do
        let (m,n)            = nn.info.dims[i]
        let (w_start, w_end) = nn.info.index[i]
        let (r_start, r_end) = nn.nn_outputs[i]
        let weights          = unflatten n m nn.data.w[w_start:w_end]
        let res              = unsafe flatten (lalg.matmul weights (unflatten m 1 retval[retval_start:retval_end]))
        let res_bias         = unsafe util.addbias res nn.data.b[b_i: b_i + n]
        let retval_i_diff    = (r_end - r_start)
        let retval           = unsafe scatter retval (retval_end..<(retval_end + retval_i_diff)) (res_bias)
      in (retval, retval_start + m, retval_end + n, b_i + n)
    in retval

   let feed_forward_batch (nn:NN) (input:[][]t) =
     map (\x -> feed_forward nn x) input


   let backprop_batch [m] (nn:NN) (input:[m][]t) (labels:[m][]t) (alpha: t) =
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
     let final_l_delta_flat = map (\x -> flatten x) final_l_delta

     let grads_w_reduced = reduce (\xr yr -> map2 (\x y -> R.((x + y))) xr yr) final_l_grads[0] final_l_grads[1:]
     let grads_b_reduced = reduce (\xr yr -> map2 (\x y -> R.((x + y))) xr yr)  final_l_delta_flat[0] final_l_delta_flat[1:]

     let tmp_w           = map (\_ -> R.(i32 0)) (0..<w_i)
     let tmp_b           = map (\_ -> R.(i32 0)) (0..<b_i)

     let grads_w         = scatter (tmp_w) (w_i-grads_w_i..<w_i) grads_w_reduced
     let grads_b         = scatter (tmp_b) (b_i-grads_b_i..<b_i) grads_b_reduced
     let l_iter = l_iter - 1
     let w_i = w_i - grads_w_i
     let b_i  = b_i - grads_b_i

     let (grads_w, grads_b, cur_l_delta, _, input_iter, w_i, b_i) =
       loop (grads_w, grads_b, final_l_delta, l_iter, input_iter, w_i, b_i)
        while l_iter > 0 do
          let next_l_delta       = final_l_delta
          let (n_l_row, n_l_col) = nn.info.dims[l_iter+1]
          let next_l_offset      = n_l_col * n_l_row
          let next_l_weights     = transpose (unflatten n_l_col n_l_row nn.data.w[w_i-next_l_offset:w_i])

          let (cur_l_row, cur_l_col) = nn.info.dims[l_iter]
          let cur_l_output           = map (\x -> x[input_iter-cur_l_col:input_iter]) input
          let cur_l_error            = map (\x -> (flatten (lalg.matmul next_l_weights x))) next_l_delta
          let cur_l_deriv            = map (\x -> act_funcs.calc_derivative x 0) cur_l_output


          let (_, prev_l_col)        = nn.info.dims[l_iter - 1]
          let cur_l_delta            = map2 (\x y -> unflatten n_l_row 1 (util.multV x y)) cur_l_error cur_l_deriv
          let input_iter             = input_iter - 1
          let prev_l_out             = map (\x -> transpose (unflatten prev_l_col 1 x[input_iter-prev_l_col:input_iter])) input
          let cur_l_grad             = map2 (\x y -> flatten (lalg.matmul x y)) cur_l_delta prev_l_out

          let grads_w_i  = cur_l_row * cur_l_col
          let grads_b_i  = cur_l_row

          let cur_l_delta_flat  = map (\x -> flatten x ) cur_l_delta
          let grads_w_reduced = reduce (\xr yr -> map2 (\x y -> R.((x + y))) xr yr) cur_l_grad[0] cur_l_grad[1:]
          let grads_b_reduced = reduce (\xr yr -> map2 (\x y -> R.((x + y))) xr yr) cur_l_delta_flat[0] cur_l_delta_flat[1:]

          let grads_w         = scatter (grads_w) (w_i-grads_w_i..<w_i) grads_w_reduced
          let grads_b         = scatter (grads_b) (b_i-grads_b_i..<b_i) grads_b_reduced

          let w_i  = w_i - grads_w_i
          let b_i  = b_i - grads_b_i
        in (grads_w, grads_b, cur_l_delta, l_iter -1, input_iter, w_i, b_i)

      let l_iter = 1
      let second_l_delta = cur_l_delta
      let (second_l_row, second_l_col )  = nn.info.dims[l_iter]
      let second_l_offset  = second_l_row * second_l_col
      let second_l_w       =
        transpose (unflatten second_l_col second_l_row nn.data.w[w_i - second_l_offset:w_i])

      let (first_l_row, first_l_col) = nn.info.dims[l_iter - 1]
      let first_l_error              = map (\x -> (flatten (lalg.matmul second_l_w x))) second_l_delta

      let first_l_output             = map (\x -> x[input_iter-first_l_col:input_iter]) input
      let first_l_deriv              = map (\x -> act_funcs.calc_derivative x 0) first_l_output

      let first_l_delta              = map2 (\x y -> unflatten second_l_row 1 (util.multV x y)) first_l_error first_l_deriv
      let first_l_input              = map (\x -> transpose (unflatten first_l_row 1 x[:first_l_row]) ) input
      let first_l_grad               = map2 (\x y -> flatten (lalg.matmul x y)) first_l_delta first_l_input

      let grads_w_i  = first_l_row * first_l_col
      let grads_b_i  = first_l_col

      let first_l_delta_flat  = map (\x -> flatten x) first_l_delta
      let grads_w_reduced = reduce (\xr yr -> map2 (\x y -> R.((x + y))) xr yr) first_l_grad[0] first_l_grad[1:]
      let grads_b_reduced = reduce (\xr yr -> map2 (\x y -> R.((x + y))) xr yr) first_l_delta_flat[0] first_l_delta_flat[1:]

      let grads_w         = scatter (grads_w) (w_i-grads_w_i..<w_i) grads_w_reduced
      let grads_b         = scatter (grads_b) (b_i-grads_b_i..<b_i) grads_b_reduced

      let grads_w'         = map (\x -> R.(alpha * x)) grads_w
      let grads_b'         = map (\x -> R.(alpha * x)) grads_b
      in (grads_w, grads_b)

   let train_batch (nn:NN) (data:[][]t) (labels:[][]t) (alpha: t)=
     let output = feed_forward_batch nn data
     let (grads_w', grads_b') = backprop_batch nn output labels alpha
     ------- Update network ------------
     let new_w               = util.subV nn.data.w grads_w'
     let new_b               = util.subV nn.data.b grads_b'
     in ({data = {w = new_w, b = new_b}, info = nn.info, nn_outputs = nn.nn_outputs }, grads_w')


}
