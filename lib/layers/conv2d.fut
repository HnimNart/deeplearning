import "../types"
import "layer_type"
import "../activations"
import "/futlib/linalg"
import "../util"


module conv2d (R:real) : layer with t = R.t
                               with input = [][][]R.t
                               with input_params = (i32,i32, i32)
                               with weights = ([][]R.t, []R.t)
                               with output  = ([][][]R.t)
                               with error_in = ([][][]R.t)
                               with error_out = ([][][]R.t)
                               with gradients = ([][][]R.t ,([][]R.t, []R.t))
                               with layer = NN ([][][]R.t) ([][]R.t,[]R.t) ([][][]R.t) ([][][]R.t) ([][][]R.t) ([][][]R.t) (R.t)

                               with act = ([]R.t -> []R.t) = {


  type t = R.t
  type input = [][][]t
  type weights  = ([][]t, []t)
  type output = [][][]t
  type garbage  = [][][]t
  type error_in = [][][]t
  type error_out = [][][]t
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
    let (x_m, x_n)      = (length input[0], length input[0,0])
    let (out_m, out_n)  = (((x_m - w_m)/ stride) + 1, ((x_n - w_n)/stride) + 1 )
    let res             = convolution input w stride (w_m, w_n) (out_m, out_n)
    let res_bias = map2 (\b' res' -> map (\x -> R.(x + b')) res') b res
    let res_act  = map (\x -> act x) res_bias
    in map (\x -> unflatten out_m out_n x) res_act


  let backward (act:act) (k:i32) (stride:i32) (l_layer:bool) ((w,b): weights) (input:input) (error:error_in) : gradients =
    if l_layer then
      (error, (w,b))
    else

  let (x_m, x_n)      = (length input[0], length input[0,0])
  let (out_m, out_n)  = (((x_m - k)/ stride) + 1, ((x_n - k)/stride) + 1 )
  let (err_m, err_n) = (length error[0], length error[0,0])

  let res  = convolution input w  1 (k,k) (out_m,out_n)
  let res_deriv = map (\x -> act x) res
  let error_flat = map (\x -> flatten x) error
  let delta = map2 (\x y -> map2 (\x' y' -> R.(x' * y')) x y) error_flat res_deriv
  -- let delta_flipped = map (\x -> reverse x) delta
  let grad_w = convolution input delta 1 (err_m, err_n) (k,k)
  let grad_b = map (\x -> R.sum x) delta

  -- Calcluate error for previous layer
  let w_flipped       = map (\x -> reverse x) w
  let delta_unflatten = map (\x -> unflatten err_m err_n x) delta
  let error_pad       = map (\x -> add_padding (k-1) x) delta_unflatten
  let error           = convolution error_pad w_flipped 1 (k,k) (x_m,x_n)
  let error_sum       = unflatten x_m x_n (map (R.sum) (transpose (map (map R.((/i32 1))) error)))
  -- let error_unflat    = map (\x -> unflatten x_m x_n x) error
  -- let error_reduce    = unflatten x_m x_n (map (R.sum)  (transpose error))
  let error' = replicate (length input) error_sum

       in (error', (grad_w,grad_b))



  let update (alpha:t) ((w,b):weights) ((wg,bg):weights) =
    let bg_scaled = map (\x -> R.(alpha * x)) bg
    let wg_scaled = map (\x -> map (\y -> R.(alpha * y)) x) wg

    let w'        = map2 (\xr yr -> map2 (\x y -> R.(x - y)) xr yr) w wg_scaled
    let b'        = map2 (\x y -> R.(x -y)) b bg_scaled
    in (w',b')

  let layer  ((filters, kernel, stride):input_params)  (act:(act,act))   =
    let w: [][]t  = (random.gen_random_array_2d ( (kernel* kernel), filters) 3)
    let b: []t    = map (\_ -> R.(i32 0)) (0..<filters)
   in
    (\w input -> (input, forward act.1 (kernel,kernel) stride w input),
     (backward act.2 kernel stride),
      update,
     (w,b))

}
