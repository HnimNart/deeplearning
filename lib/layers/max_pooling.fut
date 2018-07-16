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
  type garbage      = arr4d (i32, i32)
  type error_in     = arr4d t
  type error_out    = arr4d t
  type gradients    = (error_out, weights)
  type input_params = (i32, i32)
  type activations = ()
  type layer = max_pooling_tp t

  let max_val [m][n] (input:[m][n]t) =
    let inp_flat = flatten input
    let argmax   =  reduce (\n i -> if unsafe R.(inp_flat[n] > inp_flat[i]) then n else i) 0 (iota (length inp_flat))
    let (i,j)    = (argmax / n, argmax % n )
    in ((i,j), inp_flat[argmax])

  let empty_garbage : garbage = [[[[]]]]

  let forward ((m,n):(i32, i32)) (training:bool) (_:weights) (input:input) : (garbage, output) =
    let ixs = map (\x -> x * m) (0..<(length input[0,0,0]/m))
    let jxs = map (\x -> x * n) (0..<(length input[0,0]/n))
    let res = unsafe map (\image ->
                          map (\layer ->
                               map (\i -> map (\j -> let ((i',j'), res) = max_val layer[i:i+m,j:j+n]
                                                   in (((i + i'), (j' + j)), res))  jxs) ixs) image) input

    let index = map (\image -> map (\x -> map (\y -> map (\(is, _) -> is) y) x) image) res
    let output = map (\image ->  map (\x -> map (\y -> map (\(_, r) -> r) y) x) image) res
    let garbage = if training then index else empty_garbage
    in (index, output)

  let backward ((m,n): (i32, i32)) (_:weights) (input:garbage) (error:error_in) : gradients =
    let (l_m, l_n) = (length input[0,0], length input[0,0,0])
    let width      = (l_n *n )
    let height     = (l_m * m)
    let total_elem = (height * width)
    let index_flat = map (\image -> map (\x -> flatten x) image) input
    let offsets    = map (\image -> map (\f -> map (\(i,j) -> j + i * width) f ) image) index_flat
    let error_flat = map (\image -> map (\x -> flatten x) image) error
    let retval     = map (\_ -> R.(i32 0)) (0..<(total_elem))
    let error'     = map2 (\offsets' errors' -> map2 (\o e -> scatter  (copy retval) o  e) offsets' errors') offsets error_flat
    in (map (\image -> map (\x -> unflatten height width x) image) error', ())

  let update (_:apply_grad t) (_:weights) (_:weights) = ()

  let init ((m,n):(i32, i32)) (_:activations) (_: i32) =
    (forward (m,n),
     backward (m,n),
     update,
    (()))

}
