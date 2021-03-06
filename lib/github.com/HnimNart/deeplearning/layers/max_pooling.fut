import "../nn_types"
import "layer_type"

type^ max_pooling_2d_layer [nlayer] [input_m][input_n] [output_m][output_n] 't =
  NN ([nlayer][input_m][input_n]t) () ([nlayer][output_m][output_n]t)
     ([nlayer][output_m][output_n]i32)
     ([nlayer][output_m][output_n]t)
     ([nlayer][input_m][input_n]t)
     (apply_grad3 t)

-- | Max pooling 2d
module max_pooling_2d (R:real) : {
  type t = R.t
  val init : (nlayer: i64)
          -> (input_m: i64) -> (input_n: i64)
          -> (output_m: i64) -> (output_n: i64)
          -> max_pooling_2d_layer [nlayer] [input_m][input_n] [output_m][output_n] t
} = {

  type t = R.t

  --- Finds the maximum value given an matrix
  --- and returns the indexs and the value
  let max_val [m][n] (input:[m][n]t) : ((i64, i64), t) =
    let inp_flat = flatten input
    let argmax   =
      reduce (\n i ->
                 if R.(inp_flat[n] > inp_flat[i])
                 then n
                 else i)
                 0 (iota (length inp_flat))
    let (i,j)    = (argmax / n, argmax % n )
    in ((i,j), inp_flat[argmax])


  --- Forward propegate
  let forward [nlayer][input_m][input_n]
              (k: i64)
              (output_m: i64) (output_n: i64)
              (_training:bool)
              ()
              (input: [k][nlayer][input_m][input_n]t)
            : ([k][nlayer][output_m][output_n]i32,
               [k][nlayer][output_m][output_n]t) =

    let w_m = input_m / output_m
    let w_n = input_n / output_n
    let ixs = map (\x -> x * w_m) (0..<output_m)
    let jxs = map (\x -> x * w_n) (0..<output_n)
    let (offsets, output) =
      unzip (map (\image ->
         unzip (map (\layer ->
               unzip (map (\i ->
                      unzip (map (\j ->
                            let slice = layer[i:i+w_m, j:j+w_n]
                            let ((i',j'), res) = max_val slice
                            let offset = (input_m * (i'+i) + (j'+j))
                            in (i32.i64 offset, res)) jxs)) ixs)) image)) input)

    let cache = offsets
    in (cache, output)

  -- Back propegate by up-sample
  let backward [nlayer][output_m][output_n]
               (k: i64)
               (input_m: i64) (input_n: i64)
               (_first_layer:bool)
               _
               _
               (idx: [k][nlayer][output_m][output_n]i32)
               (error: [k][nlayer][output_m][output_n]t)
             : ([k][nlayer][input_m][input_n]t, ()) =

    let total_elem         = output_m * output_n
    let retval             = replicate (input_m*input_n) (R.i32 0)
    let idx_flat           = map (map (\arr -> flatten_to total_elem arr)) idx
    let error_flat         = map (map (\arr -> flatten_to total_elem arr)) error
    --- Write values back to their place
    let error'       =
      map2 (\ix_img err_img ->
            map2 (\i e ->
                  scatter (copy retval) (map i64.i32 i) e) ix_img err_img) idx_flat error_flat
    in (map (\image -> map (unflatten input_m input_n) image) error', ())

  let init (nlayer: i64)
           (input_m: i64) (input_n: i64)
           (output_m: i64) (output_n: i64)
         : max_pooling_2d_layer [nlayer] [input_m][input_n] [output_m][output_n] t =
    {forward  = \k -> forward k output_m output_n,
     backward = \k -> backward k input_m input_n,
     weights  = ()}

}
