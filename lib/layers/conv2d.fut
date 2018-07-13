import "../nn_types"
import "layer_type"
import "../activations"
import "/futlib/linalg"
import "../util"

module conv2d (R:real) : layer with t = R.t
                               with input = [][][][]R.t
                               with input_params = (i32,i32, i32, i32)
                               with weights = ([][]R.t, []R.t)
                               with output  = ([][][][]R.t)
                               with error_in = ([][][][]R.t)
                               with error_out = ([][][][]R.t)
                               with gradients = ([][][][]R.t ,([][]R.t, []R.t))
                               with layer = NN ([][][][]R.t) ([][]R.t,[]R.t) ([][][][]R.t) ((i32, i32, i32), [][][]R.t,[][][][]R.t) ([][][][]R.t) ([][][][]R.t) (updater ([][]R.t, []R.t))
                               with act = ([]R.t -> []R.t) = {


  type t = R.t
  type input = [][][][]t
  type weights  = ([][]t, []t)
  type output = [][][][]t
  type dims = (i32, i32, i32)
  type garbage  = (dims, [][][]t, [][][][]t)
  type error_in = [][][][]t
  type error_out = [][][][]t
  type gradients = (error_out, weights)
  type input_params = (i32 ,i32, i32, i32)

  type act = []t -> []t
  type layer = NN input weights output garbage error_in error_out (updater weights)

  module lalg   = linalg R
  module util   = utility R
  module random = normal_random_array R

  let zero_dims: dims = (0,0,0)
  let empty_garbage : garbage = (zero_dims, [[[]]], [[[[]]]])

  let add_2d_matrix (X:[][]t) (Y:[][]t) =
    map2 (\xr yr -> map2 (\x y -> R.(x +  y)) xr yr) X Y

  let calc_index (stride:i32) ((m,n):(i32, i32)) =
    let row_index = map (\i -> i * stride) (0..<m)
    let col_index = map (\i -> i * stride) (0..<n)
    in flatten (map (\i -> map (\j -> (i,j) ) row_index) col_index)

  -- let calc_index2 (stride:i32) ((m,n):(i32, i32)) =
  --   let row_index = map (\i -> i * stride) (0..<m)
  --   let col_index = map (\i -> i * stride) (0..<n)
  --   in flatten (map (\i -> map (\j -> (i,j) ) row_index) col_index)



  let add_padding (padding:i32) (X:[][]t) : [][]t =
    let height   = length X    + padding * 2
    let width    = length X[0] + padding * 2
    let tot_elem = width * height
    let index    = (flatten (map (\i -> (map (\j -> (i,j)) (0..<length X))) (0..<length X[0])))
    let offsets  = map (\(i,j) -> padding*width + padding + width * i + j) index
    let retval   = scatter (map (\_ -> R.(i32 0)) (0..<tot_elem)) (offsets) (flatten X)
    in unflatten height width retval

  let im2col (x:[][][]t) ((w_m, w_n):(i32, i32)) (idx:[](i32, i32)) : [][]t =
    unsafe transpose (map (\(i,j) ->  flatten (map (\layer -> flatten layer[i:i+w_m, j:j+w_n]) x)) idx)

  let im2col2 (X:[][][]t) ((w_m, w_n):(i32, i32)) (idx:[](i32, i32)) : [][]t =
     unsafe (map (\(i,j) ->  flatten (map (\layer -> flatten layer[i:i+w_m, j:j+w_n]) X)) idx)


  let forward (act:act) ((w_m, w_n):(i32, i32)) (stride:i32) (training:bool) ((w,b):weights) (input:input) : (garbage, output) =
    let (x_p, x_m, x_n)      = (length input[0], length input[0,0], length input[0,0,0])
    let (out_m, out_n)  = (((x_m - w_m)/ stride) + 1, ((x_n - w_n)/stride) + 1)
    let indexs          = calc_index stride (out_m, out_n)
    let image_matrix    = map (\image -> im2col image (w_m,w_n) indexs) input
    let res             = map (\image -> (lalg.matmul w image) ) image_matrix
    let res_bias        = map (\image -> map2 (\layer b' -> map (\x -> R.(x + b')) layer) image b) res
    let res_act         = map (\image -> map (\layer -> act layer ) image) res_bias
    let res_bias'       = map (\inp -> map (\x -> unflatten out_m out_n x) inp) res_bias
    let garbage         = if training then ((x_p, x_m, x_n),image_matrix,res_bias') else empty_garbage
    in (garbage, map (\inp -> map (\x -> unflatten out_m out_n x) inp) res_act)


  let flip_matrix (X:[][]t) =
    reverse (map (\x -> reverse x) X)

  let backward (act:act) (k:i32) (stride:i32) ((w,_): weights) ((dims,input0, input1):garbage) (error:error_in) : gradients =
    let (x_p, x_m, x_n) = dims
    let res_deriv       = map (\image -> map (\layer ->  map (\row -> act row) layer) image) input1
    let delta           = util.mult_matrix_4d error res_deriv

    let delta_flipped = map (\img -> map (\x -> flatten (flip_matrix x)) img) delta
    let grads_w       = map2 (\ i d -> transpose (lalg.matmul i (transpose d))) input0 delta_flipped
    let grad_w       = map (\d -> map (R.sum) (transpose d)) (transpose grads_w)

    let grads_b      = map (\img -> map (\layer -> R.sum (flatten layer) ) img) delta
    let grad_b       = map (R.sum) (transpose grads_b)

    --- Calc error for previous layer ----
    let kz = k * k
    let w_offsets = map (\i -> i * kz) (0..<x_p)
    let w_split = map (\i -> flatten ((map (\r -> reverse (r[i:i+kz])) w))) w_offsets
    let delta_padded    = map (\d -> map (\x -> add_padding (k-1) x) d) delta
    let delta_ix       = calc_index stride (x_m, x_n)
    let delta_col      = map (\d -> im2col d (k,k) delta_ix) delta_padded
    let error        = map (\x -> lalg.matmul (w_split) (x)) delta_col
    let error'         = map (\img -> map (\x -> (unflatten x_m x_n x)) img ) error

    in ( error' ,(grad_w,grad_b))

  let update (f:updater weights) (w:weights) (wg:weights) =
    f w wg

  let init ((filters, kernel, stride, depth):input_params)  (act:(act,act))  (seed: i32)  =
    let w: [][]t  = (random.gen_random_array_2d_w_scaling ((kernel* kernel * depth), filters) seed)
    let b: []t    = map (\_ -> R.(i32 0)) (0..<filters)
   in
    (forward act.1 (kernel,kernel) stride,
     backward act.2 kernel stride,
     update,
     (w,b))
}


  -- let im2col2 (X:[][][][]t) ((w_m, w_n):(i32, i32)) (idx:[](i32, i32)) : [][]t =
  --   unsafe transpose (flatten (transpose (map (\(i,j) ->  flatten ((map (\image ->  ((map (\layer -> flatten layer[i:i+w_m, j:j+w_n]) image))) X))) idx)))

    -- let (err_m, err_n) = (length error[0,0], length error[0,0, 0])
    -- let (x_p, x_m, x_n) = dims
    -- let res_deriv       = map (\image -> map (\layer -> act layer) image) input1
    -- let error_flat      = map (\err -> map (\x -> flatten x) err) error
    -- let delta           = util.mult_matrix_3d error_flat res_deriv
    -- let delta_rev       = map (\x -> map (\y -> reverse y) x) delta

    -- let grad_w_all       = map2 (\delta' input'  -> lalg.matmul (delta') (transpose input')) delta_rev input0
    -- let (grad_m, grad_n) = (length w, length w[0])
    -- let grad_w_all_flat  = map (\x -> flatten x) grad_w_all -- 2d now
    -- let grad_w           = transpose (unflatten grad_n grad_m (map (R.sum) (transpose grad_w_all_flat)))
    -- let grad_b_all       = map (\delta' ->  map (\x -> R.sum x) delta') delta
    -- let grad_b           = map (R.sum) (transpose grad_b_all)

    -- ---- Calc error for previous layer ----
    -- let w_flipped       = transpose (map (\x -> reverse x) w)
    -- let k_sz            = k * k
    -- let w_offsets       = map (\i -> i * k_sz) (0..<x_p)
    -- let w_split         = map (\x -> flatten w_flipped[x:x+k_sz] ) w_offsets
    -- let delta_unflatten = map (\d -> map (\x -> unflatten err_m err_n x) d) delta
    -- let delta_padded    = map (\d -> map (\x -> add_padding (k-1) x) d) delta_unflatten
    -- let indexs          = calc_index stride (x_m, x_n)
    -- let delta_matrix    = map (\d -> im2col d (k,k) indexs) delta_padded
    -- let error           = map (\x -> lalg.matmul (transpose x) (transpose w_split)) delta_matrix
    -- -- let tmp             = map (\x -> (transpose x)) error
    -- let tmp             = map (\x -> reverse (transpose x)) error
    -- let tmp2            = map (\x -> map (\y -> reverse y) x) tmp
    -- let error'          = map (\x' ->  map (\x -> unflatten x_m x_n x) x') tmp2
