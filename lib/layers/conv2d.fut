import "../types"
import "layer_type"
import "../activations"
import "/futlib/linalg"
import "../util"


module conv2d (R:real) : layer with t = R.t
                               with input = [][][][]R.t
                               with input_params = (i32,i32, i32)
                               with weights = ([][]R.t, []R.t)
                               with output  = ([][][][]R.t)
                               with error_in = ([][][][]R.t)
                               with error_out = ([][][][]R.t)
                               with gradients = ([][][][]R.t ,([][]R.t, []R.t))
                               with layer = NN ([][][][]R.t) ([][]R.t,[]R.t) ([][][][]R.t) ([][][][]R.t) ([][][][]R.t) ([][][][]R.t) (R.t)


                               with act = ([]R.t -> []R.t) = {


  type t = R.t
  type input = [][][][]t
  type weights  = ([][]t, []t)
  type output = [][][][]t
  type garbage  = [][][][]t
  type error_in = [][][][]t
  type error_out = [][][][]t
  type gradients = (error_out, weights)
  type input_params = (i32 ,i32, i32)

  type act = []t -> []t
  type layer = NN input weights output garbage error_in error_out t

  module lalg   = linalg R
  module util   = utility R
  module random = normal_random_array R


  let flip_matrix (X:[][]t) =
    reverse (map (\x -> reverse x) X)

  let add_3d_matrix (X:[][][]t) (Y:[][][]t) =
    map2 (\x y -> map2 (\xr yr -> map2 (\x' y' -> R.(x' +  y')) xr yr) x y) X Y

  let add_2d_matrix (X:[][]t) (Y:[][]t) =
    map2 (\xr yr -> map2 (\x y -> R.(x +  y)) xr yr) X Y

  let add_padding (padding:i32) (X:[][]t) : [][]t =
    let width = length X + padding*2
    let height = length X[0] + padding *2
    let total_elem = width * height
    let index =  (flatten (map (\i -> (map (\j -> (i,j)) (0..<length X))) (0..<length X[0])))
    let offsets = map (\(i,j) -> padding*width + padding + width * i + j) index
    let retval = scatter (map (\_ -> R.(i32 0)) (0..<total_elem)) (offsets) (flatten X)
    in unflatten height width retval

  let calc_index (stride:i32) ((m,n):(i32, i32)) =
    let row_index = map (\i -> i * stride) (0..<m)
    let col_index = map (\i -> i * stride) (0..<n)
    in flatten (map (\i -> map (\j -> (i,j) ) row_index) col_index)

  -- Convert 2d slices into a coulmn for matrix multiplication
  let im2col (X:[][]t) ((w_m, w_n):(i32, i32)) (idx:[](i32, i32))  =
    let col_matrix = unsafe transpose ((map (\(i,j) -> flatten X[i:i+w_m, j:j+w_n]) idx))
    in col_matrix

  let convolution (input: [][][]t) (w: [][]t)   =
    let y = map (\x -> lalg.matmul w x) input
    in if length y == 1 then y[0] else foldl (add_2d_matrix) y[0] y[:1]

  let forward (act:act) ((w_m, w_n):(i32, i32)) (stride:i32) ((w,b):weights) (input:input) : output =
    --- Maybe add padding here ---
    let (x_m, x_n)      = (length input[0,0], length input[0,0,0])
    let (out_m, out_n)  = (((x_m - w_m)/ stride) + 1, ((x_n - w_n)/stride) + 1 )
    let idx             = calc_index stride (out_m, out_n)
    let input_col       = map (\image -> map (\x -> im2col x (w_m, w_n) idx) image) input
    let res             = map (\x -> convolution x w) input_col
    let res_bias        = map (\res' ->  map2 (\b' r -> map (\x -> R.(x + b')) r) b res') res
    let res_act         = map (\x' -> map (\x -> act x) x') res_bias
    in map (\inp -> map (\x -> unflatten out_m out_n x) inp) res_act

  let backward (act:act) (k:i32) (stride:i32) ((w,b): weights) (input:input) (error:error_in) : gradients =
      let (x_m, x_n)      = (length input[0,0], length input[0,0,0])
      let (out_m, out_n)  = (((x_m - k)/ stride) + 1, ((x_n - k)/stride) + 1 )
      let (err_m, err_n)  = (length error[0,0], length error[0,0, 0])

      let idx        = calc_index stride (out_m, out_n)
      let input_col  = map (\image -> map (\x -> im2col x (k,k) idx) image) input
      let res        = map (\x -> convolution x w ) input_col
      let res_deriv  = map (\inpt -> map (\x -> act x) inpt) res

      let error_flat = map (\input' -> map (\x -> flatten x) input') error
      let delta = map2 (\e_inp res_inp -> map2 (\x y -> map2 (\x' y' -> R.(x' * y')) x y) e_inp res_inp) error_flat res_deriv

      let input_col_T = map (\x -> map (\x' -> transpose x') x ) input_col
      let grad_w_all = map2 (\input' delta' -> convolution input' delta') input_col_T delta
      let grad_w     = if length grad_w_all == 1 then grad_w_all[0] else foldl (add_2d_matrix) grad_w_all[0] grad_w_all[:1]
      let grad_w     = map (map R.((/i32 (length input)))) grad_w
      let grad_b_all = map (\delta' ->  map (\x -> R.sum x) delta' ) delta
      let grad_b     = if length grad_b_all == 1 then grad_b_all[0] else map (R.sum) (transpose grad_b_all)
      let grad_b     = map (R.((/i32 (length input)))) grad_b

      -- Calcluate error for previous layer
      let w_flipped       = map (\x -> reverse x) w
      let delta_unflatten = map (\d -> map (\x -> unflatten err_m err_n x) d) delta
      let error_pad       = map (\d -> map (\x -> add_padding (k-1) x) d) delta_unflatten
      let idx             = calc_index stride (x_m, x_n)
      let error_col       = map (\d -> map (\x -> im2col x (k,k) idx ) d) error_pad
      let error           = map (\d -> convolution d w_flipped ) error_col
      let error_sum       = map (\x -> unflatten x_m x_n (map (R.sum) (transpose x))) error
      -- let error_sum       = map (\x -> unflatten x_m x_n (map (R.sum) (transpose (map (map R.((/i32 100))) x) ))) error
      let error'          = map (\x -> replicate (length input[0]) x) error_sum
      in (error' , (grad_w, grad_b))

  let update (alpha:t) ((w,b):weights) ((wg,bg):weights) =
    let wg_scaled = map (\x -> map (\y -> R.((alpha) * y)) x) wg
    let bg_scaled = map (\x -> R.((alpha) * x)) bg
    let w'        = map2 (\xr yr -> map2 (\x y -> R.(x - y)) xr yr) w wg_scaled
    let b'        = map2 (\x y -> R.(x -y)) b bg_scaled
    in (w',b')

  let init ((filters, kernel, stride):input_params)  (act:(act,act))  (seed: i32)  =
    let w: [][]t  = (random.gen_random_array_2d_w_scaling_conv ( (kernel* kernel), filters) seed)
    let b: []t    = map (\_ -> R.(i32 0)) (0..<filters)
   in
    (\w input -> (input, forward act.1 (kernel,kernel) stride w input),
     (backward act.2 kernel stride),
      update,
     (w,b))
}
