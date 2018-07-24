import "layer_type"
import "../nn_types"
import "../util"
import "../random_gen"
import "/futlib/linalg"

module conv2d (R:real) : layer with t = R.t
                               with input_params = (i32,i32, i32, i32)
                               with activations  = activation_func ([]R.t)
                               with input        = arr4d  R.t
                               with weights      = (arr2d R.t,arr1d R.t)
                               with output       = arr4d R.t
                               with cache        = (dims3d, arr3d R.t, arr4d R.t)
                               with error_in     = arr4d R.t
                               with error_out    = arr4d R.t = {

  type t = R.t
  type input        = arr4d  t
  type weights      = std_weights t
  type output       = arr4d t
  type dims         = dims3d
  type cache        = (dims, arr3d t, arr4d t)
  type error_in     = arr4d t
  type error_out    = arr4d t
  type b_output     = (error_out, weights)

  type input_params = (i32 ,i32, i32, i32)
  type activations  = activation_func ([]R.t)

  module lalg   = linalg R
  module util   = utility R
  module random = normal_random_array R

  let zero_dims: dims = (0,0,0)
  let empty_cache: cache = (zero_dims, [[[]]], [[[[]]]])
  let empty_error: error_out = [[[[]]]]

  -- Calculate offsets in img, given window size and stride
  let calc_img_offsets (stride:i32) ((m,n):(i32, i32)): [](i32, i32) =
    let row_offsets = map (\i -> i * stride) (0..<m)
    let col_offsets = map (\i -> i * stride) (0..<n)
    in flatten (map (\i -> map (\j -> (i,j) ) row_offsets) col_offsets)

  --- Add zero padding around a 2D img
  let add_padding [m][n] (padding:i32) (X:[m][n]t) : arr2d t =
    let (output_m , output_n)  = (m + padding * 2, n + padding * 2)
    let tot_elem               = output_m * output_n

    let index    =
      flatten (map (\i -> (map (\j -> (i,j)) (0..<m))) (0..<n))
    let offsets  =
      map (\(i,j) -> padding*output_n + padding + output_m * i + j) index
    let retval   =
      scatter (map (\_ -> R.(i32 0)) (0..<tot_elem)) (offsets) (flatten X)
    in unflatten output_m output_n retval

    --- Transforms img convolution to column matrix
  let im2col (x:arr3d t)
             ((w_m, w_n):(i32, i32))
             (idx:arr1d  (i32, i32)) : arr2d t =
    unsafe transpose (map (\(i,j) ->
                          flatten (map (\layer ->
                                       flatten layer[i:i+w_m, j:j+w_n]) x)) idx)

  let forward (act:[]t -> []t)
              ((w_m, w_n):(i32, i32))
              (stride:i32)
              (training:bool)
              ((w,b):weights)
              (input:input) : (cache, output) =

    let (x_p, x_m, x_n) = (length input[0], length input[0,0], length input[0,0,0])
    let (out_m, out_n)  = (((x_m - w_m)/ stride) + 1, ((x_n - w_n)/stride) + 1)
    let img_offsets     = calc_img_offsets stride (out_m, out_n)
    let image_matrix    = map (\image -> im2col image (w_m,w_n) img_offsets) input
    let res             = map (\image -> (lalg.matmul w image) ) image_matrix
    let res_bias        =
      map (\image -> map2 (\layer b' -> map (\x -> R.(x + b')) layer) image b) res

    let res_act         = map (\image ->
                               map (\layer -> act layer ) image) res_bias
    let cache           = if training then
                            let res_bias' =
                              map (\inp ->
                                   map (\x ->
                                        unflatten out_m out_n x) inp) res_bias
                            in ((x_p, x_m, x_n), image_matrix, res_bias')
                          else empty_cache
    let output = map (\inp -> map (\x -> unflatten out_m out_n x) inp) res_act
    in (cache, output)


  let backward (act:[]t->[]t)
               (k:i32)
               (stride:i32)
               (first_layer:bool)
               ((w,_): weights)
               ((dims, img_matrix, res_bias):cache)
               (error:error_in) : b_output =

    let (x_p, x_m, x_n) = dims
    let res_deriv       = map (\image ->
                               map (\layer ->
                                    map (\row -> act row) layer) image) res_bias
    let delta           = util.mult_matrix_4d error res_deriv

    let delta_flat      = map (\img -> map (\layer -> flatten layer) img) delta
    let grads_w         =
      map2 (\img d ->
            transpose (lalg.matmul img (transpose d))) img_matrix delta_flat
    let grad_w          = map (\d -> map (R.sum) (transpose d)) (transpose grads_w)

    let grads_b         =
      map (\img -> map (\layer -> R.sum (flatten layer) ) img) delta
    let grad_b          = map (R.sum) (transpose grads_b)

    --- Calc error for previous layer ----
    let error' =
      if first_layer
      then
        empty_error
      else
        let filter_sz    = k * k
        let w_offsets    = map (\i -> i * filter_sz) (0..<x_p)
        let w_flipped    =
          unsafe map (\i ->
                      flatten ((map (\r ->
                                     reverse (r[i:i+filter_sz])) w))) w_offsets
        let delta_padded  =
          map (\delta' -> map (\x -> add_padding (k-1) x) delta') delta
        let delta_offsets = calc_img_offsets stride (x_m, x_n)
        let delta_matrix  =
          map (\delta' -> im2col delta' (k,k) delta_offsets) delta_padded
        let error         = map (\delta' -> lalg.matmul w_flipped delta') delta_matrix
        in map (\img -> map (\x -> (unflatten x_m x_n x)) img ) error

    in (error' , (grad_w,grad_b))

  let update (f:apply_grad t) (w:weights) (wg:weights) =
    f w wg

  let init ((filters, kernel, stride, depth):input_params)
           (act:activations)
           (seed: i32)  =
    let w: arr2d t =
      random.gen_random_array_2d_xavier_uni ((kernel* kernel * depth), filters) seed
    let b: arr1d t = replicate filters R.(i32 0)
    in
    {forward  = forward act.f (kernel,kernel) stride,
     backward = backward act.fd kernel stride,
     update   = update,
     weights  = (w,b)}
}
