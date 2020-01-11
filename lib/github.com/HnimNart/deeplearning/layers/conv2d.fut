import "layer_type"
import "../nn_types"
import "../util"
import "../weight_init"
import "../../../diku-dk/linalg/linalg"

-- | 2D convolutional layer
-- Uses GEMM method to perform convolution operation
module conv2d (R:real) : layer_type with t = R.t
                                    with input_params = (i32,i32, i32, i32)
                                    with activations  = activation_func ([]R.t)
                                    with input        = arr4d R.t
                                    with output       = arr4d R.t
                                    with error_in     = arr4d R.t
                                    with error_out    = arr4d R.t = {

  type t = R.t
  -- type input        = arr4d  t
  -- type weights      = std_weights t
  -- type output       = arr4d t
  -- type dims         = dims3d
  -- type cache        = (dims, arr3d t, arr4d t)
  -- type error_in     = arr4d t
  -- type error_out    = arr4d t
  -- type b_output     = (error_out, weights)

  -- type input_params = (i32 ,i32, i32, i32)
  -- type activations  = activation_func ([]R.t)

  module lalg   = mk_linalg R
  module util   = utility R
  module w_init = weight_initializer R

  let zero_dims = (0,0,0)

  -- Calculate offsets in img, given window size and stride
  let calc_img_offsets (mn: i32) (stride:i32) ((m,n):(i32, i32)): [mn](i32, i32) =
    let row_offsets = map (\i -> i * stride) (0..<m)
    let col_offsets = map (\i -> i * stride) (0..<n)
    in flatten (map (\i -> map (\j -> (i,j) ) row_offsets) col_offsets)
       : [mn](i32,i32)

  --- Add zero padding around a 2D img
  let add_padding [m][n] (padding:i32) (output_m: i32) (output_n: i32) (X:[m][n]t)
                       : [output_m][output_n]t =
--    let (output_m , output_n)  = (m + padding * 2, n + padding * 2)
    let tot_elem               = output_m * output_n

    let mn = m * n
    let flatten_mn 'a (arr: [][]a) = flatten arr : [mn]a
    let index    =
      flatten_mn (map (\i -> (map (\j -> (i,j)) (0..<m))) (0..<n))
    let offsets  =
      map (\(i,j) -> padding*output_n + padding + output_m * i + j) index
    let retval   =
      scatter (map (\_ -> R.(i32 0)) (0..<tot_elem)) offsets (flatten_mn X)
    in unflatten output_m output_n retval

  -- Transforms 3D imgage to column matrix
  -- for convolutional op
  let im2col [p][m][n] [l]
             (x: [p][m][n]t)
             ((w_m, w_n):(i32, i32))
             (pwtotal: i32)
             (idx: [l](i32, i32)) : [pwtotal][l]t =
    let wtotal = w_m * w_n
    in unsafe transpose (map (\(i,j) ->
                                flatten (map (\layer ->
                                                flatten layer[i:i+w_m, j:j+w_n]
                                                : [wtotal]t)
                                             x)
                                : [pwtotal]t)
                             idx)

  let forward [k][pwtotal]
              (p: i32) (m: i32) (n: i32)
              (out_m: i32) (out_n: i32)
              (out_mn: i32)
              (act: [out_mn]t -> [out_mn]t)
              ((w_m, w_n): (i32, i32))
              (stride:i32)
              (training:bool)
              ((w,b): std_weights [p][pwtotal] [p] t)
              (input: [k][p][m][n]t)
            : ([k]([pwtotal][out_mn]t, [p][out_m][out_n]t),
               [k][p][out_m][out_n]t) =

    let img_offsets     = calc_img_offsets out_mn stride (out_m, out_n)
    let image_matrix    = map (\image -> im2col image (w_m,w_n) pwtotal img_offsets) input
    let res : [k][p][out_mn]t = map (lalg.matmul w) image_matrix
    let res_bias        =
      map (\image -> map2 (\layer b' -> map (\x -> R.(x + b')) layer) image b) res

    let res_act         = map (map act) res_bias
    let cache           = let res_bias' =
                            map (\inp ->
                                   map (\x ->
                                          unflatten out_m out_n x) inp) res_bias
                          in zip image_matrix res_bias'
    let output = map (\inp -> map (\x -> unflatten out_m out_n x) inp) res_act
    in (cache, output)


  let backward [k][pwtotal]
               (p: i32) (m: i32) (n: i32)
               (out_m: i32) (out_n: i32)
               (out_mn: i32)
               (act: [out_n]t -> [out_n]t)
               (filter_d:i32)
               (stride:i32)
               (first_layer:bool)
               (apply_grads: apply_grad3 t)
               ((w,b): std_weights [p][pwtotal] [p] t)
               ((cache: [k]([pwtotal][out_mn]t, [p][out_m][out_n]t)))
               (error: [k][p][out_m][out_n]t)
             : (bool, std_weights [p][pwtotal] [p] t) =

    let (img_matrix, res_bias) = unzip cache
    let res_deriv       = map (\image ->
                               map (\layer ->
                                    map (\row -> act row) layer) image) res_bias
    let delta           = util.hadamard_prod_4d error res_deriv

    let delta_flat      = map (\img -> map (\layer -> flatten layer : [out_mn]t) img) delta
    let grads_w         =
      map2 (\img d ->
            transpose (lalg.matmul img (transpose d))) img_matrix delta_flat

    let grad_w    = map (\d -> map (R.sum) (transpose d)) (transpose grads_w)

    let grads_b   =
      map (\img -> map (\layer -> R.sum (flatten layer) ) img) delta
    let grad_b    = map (R.sum) (transpose grads_b)

    let (w', b')  = apply_grads p pwtotal (w,b) (grad_w, grad_b)

    --- Calc error for previous layer ----
    let error' =
      let filter_sz    = filter_d * filter_d
      let w_offsets    = map (\i -> i * filter_sz) (0..<p)
      let w_flipped    =
        unsafe map (\i ->
                    flatten ((map (\r ->
                                   reverse (r[i:i+filter_sz]) : [filter_sz]t) w))) w_offsets
      let delta_padded  =
        map (\delta' -> map (\x -> add_padding (filter_d-1) x) delta') delta
      let delta_offsets = calc_img_offsets stride (m, n)
      let delta_matrix  =
        map (\delta' -> im2col delta' (filter_d,filter_d) delta_offsets) delta_padded
      let error         = map (\delta' -> lalg.matmul w_flipped delta') delta_matrix
      in map (\img -> map (\x -> (unflatten m n x)) img ) error
    in (error' , (w',b'))


  let init ((filters, k , stride, depth):input_params)
           (act:activations)
           (seed: i32)  =
    let w: arr2d t =
      w_init.gen_random_array_2d_xavier_uni ((k * k * depth), filters) seed
    let b: arr1d t = replicate filters R.(i32 0)
    in
    {forward  = forward act.f (k,k) stride,
     backward = backward act.fd k stride,
     weights  = (w,b)}
}
