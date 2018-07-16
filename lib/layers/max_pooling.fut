import "../nn_types"
import "layer_type"
import "/futlib/linalg"
import "../util"



module max_pooling_2d (R:real) : layer with t = R.t
                                       with input_params = (i32 , i32)
                                       with activations = ()
                                       with layer = max_pooling_tp R.t = {

  type t = R.t
  type input        = arr4d t
  type weights      = ()
  type output       = arr4d t
  type garbage      = arr4d (i32)
  type error_in     = arr4d t
  type error_out    = arr4d t
  type gradients    = (error_out, weights)
  type input_params = (i32, i32)
  type activations  = ()
  type layer = max_pooling_tp t

  --- Finds the maximum value given an matrix
  --- and returns the indexs and the value
  let max_val [m][n] (input:[m][n]t) =
    let inp_flat = flatten input
    let argmax   = unsafe reduce (\n i ->
                                  if R.(inp_flat[n] > inp_flat[i])
                                  then n
                                  else i) 0 (iota (length inp_flat))
    let (i,j)    = (argmax / n, argmax % n )
    in ((i,j), inp_flat[argmax])

  let empty_garbage : garbage = [[[[]]]]

  let forward ((m,n):(i32, i32)) (training:bool) (_:weights) (input:input) : (garbage, output) =
    let (input_m, input_n)    = (length input[0,0], length input[0,0,0])
    let (output_m, output_n)  = (input_m/m, input_n/n)
    let ixs = map (\x -> x * m) (0..<output_m)
    let jxs = map (\x -> x * n) (0..<output_n)
    let res = unsafe map (\image ->
                          map (\layer ->
                               map (\i ->
                                    map (\j -> let ((i',j'), res) = max_val layer[i:i+m,j:j+n]
                                               let offset = (input_m * (i'+i) + (j'+j))
                                               in (offset, res)) jxs) ixs) image) input

    let index   = map (\image -> map (\x -> map (\y -> map (\(is, _) -> is) y) x) image) res
    let output  = map (\image -> map (\x -> map (\y -> map (\(_, r) -> r) y) x) image) res
    let garbage = if training then index else empty_garbage
   in (garbage, output)

  let backward ((m,n): (i32, i32)) (_:weights) (indexs:garbage) (error:error_in) : gradients =
    --- Recreate dimensions
    let (l_m, l_n) = (length indexs[0,0], length indexs[0,0,0])
    let height     = (l_m*m)
    let width      = (l_n*n)
    let total_elem = (height * width)

    let index_flat = map (\image -> map (\layer -> flatten layer) image) indexs
    let error_flat = map (\image -> map (\layer -> flatten layer) image) error
    let retval     = map (\_ -> R.(i32 0)) (0..<(total_elem))
    let error'     =
      map2 (\ix_img err_img -> map2 (\i e -> scatter (copy retval) i e) ix_img err_img) index_flat error_flat
    in (map (\image -> map (\x -> unflatten height width x) image) error', ())

  let update (_:apply_grad t) (_:weights) (_:weights) = ()

  let init ((m,n):(i32, i32)) (_:activations) (_: i32) =
    (forward (m,n),
     backward (m,n),
     update,
     ())

}
