import "layer_type"
import "../nn_types"
import "../util"
import "../weight_init"
import "../../../diku-dk/linalg/linalg"

type^ conv2d_layer [p][m][n] [filter_d] [filters] [out_m] [out_n] 't =
  NN ([p][m][n]t) ([filters][p][filter_d][filter_d]t, [filters]t) ([filters][out_m][out_n]t)
     ([p][filter_d][filter_d][out_m][out_n]t, [filters][out_m][out_n]t)
     ([filters][out_m][out_n]t)
     ([p][m][n]t)
     (apply_grad3 t)

-- | 2D convolutional layer
-- Uses GEMM method to perform convolution operation
module conv2d (R:real) : {
  type t = R.t
  val init :
       (p: i64) -> (m: i64) -> (n: i64) ->
       (filter_d: i64) -> (stride: i32) -> (filters: i64) ->
       (out_m: i64) -> (out_n: i64)
    -> ((d: i64) -> activation_func ([d]t))
    -> i32
    -> conv2d_layer [p][m][n] [filter_d] [filters] [out_m] [out_n] t
  } = {

  type t = R.t

  module lalg   = mk_linalg R
  module util   = utility R
  module w_init = weight_initializer R

  -- Calculate offsets in img, given window size and stride
  let calc_img_offsets (mn: i64) (stride:i32) ((m,n):(i64, i64)): [mn](i64, i64) =
    let row_offsets = map (\i -> i * i64.i32 stride) (0..<m)
    let col_offsets = map (\i -> i * i64.i32 stride) (0..<n)
    in flatten_to mn (map (\i -> map (\j -> (i,j) ) row_offsets) col_offsets)

  --- Add zero padding around a 2D img
  let add_padding [m][n] (padding:i64) (output_m: i64) (output_n: i64) (X:[m][n]t)
                       : [output_m][output_n]t =
    let tot_elem = output_m * output_n
    let mn = m * n
    let index =
      flatten_to mn (map (\i -> (map (\j -> (i,j)) (0..<m))) (0..<n))
    let offsets  =
      map (\(i,j) -> padding*output_n + padding + output_m * i + j) index
    let retval =
      scatter (map (\_ -> R.(i32 0)) (0..<tot_elem)) offsets (flatten_to mn X)
    in unflatten output_m output_n retval

  -- Transforms 3D imgage to column matrix
  -- for convolutional op
  let im2col [p][m][n] [mn]
             (x: [p][m][n]t)
             ((w_m, w_n):(i64, i64))
             (pwtotal: i64)
             (idx: [mn](i64, i64)) : [pwtotal][mn]t =
    let wtotal = w_m * w_n
    in transpose (map (\(i,j) ->
                         flatten_to pwtotal
                                    (map (\layer ->
                                            flatten_to wtotal layer[i:i+w_m, j:j+w_n])
                                         x))
                      idx)

  let forward [filter_d][filters]
              (k: i64) (p: i64) (m: i64) (n: i64)
              (out_m: i64) (out_n: i64)
              (out_mn: i64)
              (act: [out_mn]t -> [out_mn]t)
              ((w_m, w_n): (i64, i64))
              (stride:i32)
              (_training:bool)
              ((w,b): ([filters][p][filter_d][filter_d]t, [filters]t))
              (input: [k][p][m][n]t)
            : ([k]([p][filter_d][filter_d][out_m][out_n]t, [filters][out_m][out_n]t),
               [k][filters][out_m][out_n]t) =

    let pwtotal         = p * filter_d * filter_d
    let w               = map (\w' -> flatten_3d w' :> [pwtotal]t) w
    let img_offsets     = calc_img_offsets out_mn stride (out_m, out_n)
    let image_matrix    = map (\image -> im2col image (w_m,w_n) pwtotal img_offsets) input
    let res = map (lalg.matmul w) image_matrix
    let res_bias        =
      map (\image -> map2 (\layer b' -> map (\x -> R.(x + b')) layer) image b) res

    let res_act         = map (map act) res_bias
    let cache           = let image_matrix' =
                            image_matrix
                            |> map (map (unflatten out_m out_n))
                            |> map (unflatten_3d p filter_d filter_d)
                          let res_bias' =
                            map (map (unflatten out_m out_n)) res_bias
                          in zip image_matrix' res_bias'
    let output = map (\inp -> map (\x -> unflatten out_m out_n x) inp) res_act
    in (cache, output)


  let backward [filter_d][filters]
               (k: i64) (p: i64) (m: i64) (n: i64)
               (out_m: i64) (out_n: i64)
               (out_mn: i64)
               (act: [out_n]t -> [out_n]t)
               (stride:i32)
               (_first_layer:bool)
               (apply_grads: apply_grad3 t)
               ((w,b): ([filters][p][filter_d][filter_d]t, [filters]t))
               ((cache: [k]([p][filter_d][filter_d][out_m][out_n]t, [filters][out_m][out_n]t)))
               (error: [k][filters][out_m][out_n]t)
             : ([k][p][m][n]t,
                ([filters][p][filter_d][filter_d]t, [filters]t)) =

    let pwtotal         = p * filter_d * filter_d
    let (img_matrix_nested, res_bias) = unzip cache
    let w               = map (\w' -> flatten_3d w' :> [pwtotal]t) w
    let img_matrix      = img_matrix_nested
                          |> map (map (map (map (\layer -> flatten_to out_mn layer))))
                          |> map (\img -> flatten_3d img :> [pwtotal][out_mn]t)
    let res_deriv       = map (\image ->
                               map (\layer ->
                                    map (\row -> act row) layer) image) res_bias
    let delta           = util.hadamard_prod_4d error res_deriv

    let delta_flat      = map (\img -> map (\layer -> flatten_to out_mn layer) img) delta
    let grads_w         =
      map2 (\img d ->
            transpose (lalg.matmul img (transpose d))) img_matrix delta_flat

    let grad_w    = map (\d -> map (R.sum) (transpose d)) (transpose grads_w)

    let grads_b   =
      map (\img -> map (\layer -> R.sum (flatten layer) ) img) delta
    let grad_b    = map (R.sum) (transpose grads_b)

    let (w', b')  = apply_grads filters pwtotal (w,b) (grad_w, grad_b)

    --- Calc error for previous layer ----
    let error' =
      let filter_sz    = filter_d * filter_d
      let w_offsets    = map (\i -> i * filter_sz) (0..<p)
      let filters_filter_sz  = filters * filter_sz
      let w_flipped    =
        map (\i -> flatten_to filters_filter_sz
                   (map (\r -> reverse r[i:i+filter_sz]
                               :> [filter_sz]t) w)) w_offsets
      let out_m_padded = out_m+(filter_d-1)*2
      let out_n_padded = out_n+(filter_d-1)*2
      let delta_padded  =
        map (\delta' -> map (\x -> add_padding (filter_d-1) out_m_padded out_n_padded x)
                            delta')
            delta
      let mn = m*n
      let delta_offsets = calc_img_offsets mn stride (m, n)
      let delta_matrix  =
        map (\delta' -> im2col delta' (filter_d,filter_d) filters_filter_sz delta_offsets) delta_padded
      let error         = map (\delta' -> lalg.matmul w_flipped delta') delta_matrix
      in map (\img -> map (\x -> (unflatten m n x)) img ) error
    in (error',
        (map (unflatten_3d p filter_d filter_d) w',
         b'))


  let init (p: i64) (m: i64) (n: i64)
           (filter_d: i64) (stride: i32)
           (filters: i64) (out_m: i64) (out_n: i64)
           (act: (d: i64) -> activation_func ([d]t))
           (seed: i32) : conv2d_layer [p][m][n] [filter_d] [filters] [out_m] [out_n] t =
    let pwtotal = p * filter_d * filter_d
    let out_mn = out_m * out_n
    let w = w_init.gen_random_array_2d_xavier_uni pwtotal filters seed
    let b = replicate filters R.(i32 0)
    in {forward  = \k -> forward k p m n out_m out_n out_mn (act out_mn).f (filter_d, filter_d) stride,
        backward = \k -> backward k p m n out_m out_n out_mn (act out_n).fd stride,
        weights  = (map (unflatten_3d p filter_d filter_d) w,
                    b)}
}
