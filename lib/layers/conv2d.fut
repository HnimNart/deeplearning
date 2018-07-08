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

  -- Convert 2d slices into a coulmn for matrix multiplication
  let im2col (X:[][]t) (stride:i32) ((w_m, w_n):(i32, i32)) ((out_m, out_n):(i32, i32)) =
    let col_index = map (\i -> i * stride) (0..<out_n)
    let row_index = map (\i -> i * stride) (0..<out_m)
    let indexes   = flatten (map (\i -> map (\j -> (i,j) ) row_index) col_index)
    let col_matrix = unsafe transpose ((map (\(i,j) -> flatten X[i:i+w_m, j:j+w_n]) indexes))
    in col_matrix

  let convolution (input: [][][]t) (w: [][]t) (stride:i32) ((w_m,w_n):(i32, i32)) ((out_m, out_n):(i32, i32))  =
    let m = map (\x -> im2col x stride (w_m,w_n) (out_m, out_n) ) input
    let y = map (\x -> lalg.matmul w x)  m
    in if length y == 1 then y[0] else reduce (add_2d_matrix) y[0] y[:1]

  let forward (act:act) ((w_m, w_n):(i32, i32)) (stride:i32) ((w,b):weights) (input:input) : output =
    --- Maybe add padding here ---
    let (x_m, x_n)      = (length input[0,0], length input[0,0,0])
    let (out_m, out_n)  = (((x_m - w_m)/ stride) + 1, ((x_n - w_n)/stride) + 1 )
    let res             = map (\x -> convolution x w stride (w_m, w_n) (out_m, out_n)) input
    let res_bias        = map (\res' ->  map2 (\b' r -> map (\x -> R.(x + b')) r) b res') res
    let res_act         = map (\x' -> map (\x -> act x) x') res_bias
    in map (\inp -> map (\x -> unflatten out_m out_n x) inp) res_act


  let backward (act:act) (k:i32) (stride:i32) (l_layer:bool) ((w,b): weights) (input:input) (error:error_in) : gradients =
    if l_layer then
      (error, (w,b))
    else

  let (x_m, x_n)      = (length input[0,0], length input[0,0,0])
  let (out_m, out_n)  = (((x_m - k)/ stride) + 1, ((x_n - k)/stride) + 1 )
  let (err_m, err_n) = (length error[0,0], length error[0,0, 0])

  let res  = map (\x -> convolution x w 1 (k,k) (out_m,out_n)) input
  let res_deriv = map (\inpt -> map (\x -> act x) inpt) res

  let error_flat = map (\input' -> map (\x -> flatten x) input') error
  let delta = map2 (\e_inp res_inp -> map2 (\x y -> map2 (\x' y' -> R.(x' * y')) x y) e_inp res_inp) error_flat res_deriv

  let grad_w_all = map2 (\input' delta' -> convolution input' delta' stride (err_m, err_n) (k,k)) input delta
  let grad_w     = if length grad_w_all == 1 then grad_w_all[0] else reduce (add_2d_matrix) grad_w_all[0] grad_w_all[:1]
  let grad_b_all = map (\delta' ->  map (\x -> R.sum x) delta' ) delta
  let grad_b     = if length grad_b_all == 1 then grad_b_all[0] else map (R.sum) (transpose grad_b_all)

  -- -- Calcluate error for previous layer
  let w_flipped       = map (\x -> reverse x) w
  let delta_unflatten = map (\d -> map (\x -> unflatten err_m err_n x) d) delta
  let error_pad       = map (\d -> map (\x -> add_padding (k-1) x) d) delta_unflatten
  let error           = map (\d -> convolution d w_flipped 1 (k,k) (x_m,x_n)) error_pad
  let error_sum       = map (\x -> unflatten x_m x_n (map (R.sum) (transpose (map (map R.((/i32 1))) x)))) error
  let error' = map (\x -> replicate (length input)  x )error_sum

   in (error' , (grad_w, grad_b))
       -- in (error', (grad_w,grad_b))



  let update (alpha:t) ((w,b):weights) ((wg,bg):weights) =
    let bg_scaled = map (\x -> R.(alpha * x)) bg
    let wg_scaled = map (\x -> map (\y -> R.(alpha * y)) x) wg

    let w'        = map2 (\xr yr -> map2 (\x y -> R.(x - y)) xr yr) w wg_scaled
    let b'        = map2 (\x y -> R.(x -y)) b bg_scaled
    in (w',b')

  let layer  ((filters, kernel, stride):input_params)  (act:(act,act))   =
    let w: [][]t  = (random.gen_random_array_2d ( (kernel* kernel), filters) 1)
    let b: []t    = map (\_ -> R.(i32 0)) (0..<filters)
   in
    (\w input -> (input, forward act.1 (kernel,kernel) stride w input),
     (backward act.2 kernel stride),
      update,
     (w,b))

}
