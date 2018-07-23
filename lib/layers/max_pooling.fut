import "../nn_types"
import "layer_type"
import "/futlib/linalg"
import "../util"



module max_pooling_2d (R:real) : layer with t = R.t
                                       with input_params = (i32 , i32)
                                       with activations = ()
                                       with input        = arr4d R.t
                                       with weights      = ()
                                       with output       = arr4d R.t
                                       with cache        = arr4d (i32)
                                       with error_in     = arr4d R.t
                                       with error_out    = arr4d R.t = {

  type t = R.t
  type input        = arr4d t
  type weights      = ()
  type output       = arr4d t
  type cache        = arr4d (i32)
  type error_in     = arr4d t
  type error_out    = arr4d t
  type b_output     = (error_out, weights)

  type input_params = (i32, i32)
  type activations  = ()
  type max_pool     = NN input weights output cache error_in error_out (apply_grad t)

  let empty_cache : cache = [[[[]]]]
  let empty_error : error_out = [[[[]]]]

  --- Finds the maximum value given an matrix
  --- and returns the indexs and the value
  let max_val [m][n] (input:[m][n]t) : ((i32, i32), t) =
    let inp_flat = flatten input
    let argmax   = unsafe reduce (\n i ->
                                  if R.(inp_flat[n] > inp_flat[i])
                                  then n
                                  else i) 0 (iota (length inp_flat))
    let (i,j)    = (argmax / n, argmax % n )
    in ((i,j), inp_flat[argmax])


  let forward ((w_m,w_n):(i32, i32)) (training:bool) (_:weights) (input:input) : (cache, output) =
    let (input_m, input_n)    = (length input[0,0], length input[0,0,0])
    let (output_m, output_n)  = (input_m/w_m, input_n/w_n)
    let ixs = map (\x -> x * w_m) (0..<output_m)
    let jxs = map (\x -> x * w_n) (0..<output_n)
    let res = unsafe map (\image ->
                          map (\layer ->
                               map (\i ->
                                    map (\j -> let ((i',j'), res) = max_val layer[i:i+w_m,j:j+w_n]
                                               let offset = (input_m * (i'+i) + (j'+j))
                                               in (offset, res)) jxs) ixs) image) input

    let cache   = if training
                  then map (\image ->
                            map (\layer ->
                                 map (\row ->
                                      map (\(is, _) -> is) row) layer) image) res
                  else empty_cache
    let output  = map (\image -> map (\x -> map (\y -> map (\(_, r) -> r) y) x) image) res
    in (cache, output)

  let backward ((m,n): (i32, i32)) (first_layer:bool) (_:weights) (indexs:cache) (error:error_in): (error_out, weights) =
    if first_layer then
      (empty_error, ())
    else
    --- Recreate dimensions
      let (l_m, l_n) = (length indexs[0,0], length indexs[0,0,0])
      let height     = (l_m*m)
      let width      = (l_n*n)
      let total_elem = (height * width)
      let retval     = map (\_ -> R.(i32 0)) (0..<(total_elem))
      let index_flat = map (\image -> map (\layer -> flatten layer) image) indexs
      let error_flat = map (\image -> map (\layer -> flatten layer) image) error
      let error'     =
      map2 (\ix_img err_img -> map2 (\i e -> scatter (copy retval) i e) ix_img err_img) index_flat error_flat
    in (map (\image -> map (\x -> unflatten height width x) image) error', ())

  let update (_:apply_grad t) (_:weights) (_:weights) : weights = ()

  let init ((m,n):(i32, i32)) (_:activations) (_: i32) : max_pool =
    { forward = forward (m,n),
     backward = backward (m,n),
     update = update,
     weights = ()}

}
