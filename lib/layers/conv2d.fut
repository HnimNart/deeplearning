import "../types"
import "layer_type"
import "../activations"
import "/futlib/linalg"
import "../util"


module conv2d (R:real) : layer with t = R.t
                               with input = [][][]R.t
                               with input_params = (i32,i32)
                               with weights = ([][][]R.t, []R.t)
                               with output  = ([][][]R.t)
                               with error_in = ([][][]R.t)
                               with error_out = ([][][]R.t)
                               with gradients = ([][][]R.t ,([][][]R.t, []R.t))
                               with layer = NN ([][][]R.t) ([][][]R.t,[]R.t) ([][][]R.t) ([][][]R.t) ([][][]R.t) ([][][]R.t) (R.t)

                               with act = ([][]R.t -> [][]R.t) = {


  type t = R.t
  type input = [][][]t
  type weights  = ([][][]t, []t)
  type output = [][][]t
  type garbage  = [][][]t
  type error_in = [][][]t
  type error_out = [][][]t
  type gradients = (error_out, weights)
  type input_params = (i32 ,i32)

  type act = [][]t -> [][]t
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


  let dot_prod_matrix (input: [][]t) (w:[][]t)  =
    R.(sum (flatten (util.mult_matrix input w)))

  let convolve (input: [][][]t) (w: [][]t) (b:t) =
    let (m,n) = (length input[0], length input[0,0])
    let (wm, wn) = (length w, length w[0])
    let (m_d, n_d) = (m-wm+1, n -wn+1)
    let res =
      unsafe map (\layer ->
                  map (\i ->
                       map (\j -> let res = dot_prod_matrix layer[i:i+wm,j:j+wn] w
                                  in R.(res + b)) (0..<n_d)) (0..<m_d)) input
    let ne = replicate (m_d) (replicate (n_d) R.(i32 0))
    in reduce (util.add_matrix) ne res

  let forward (act:act) ((w,b):weights) (input:input) : output =
    let res = map2 (\w' b' -> convolve input w' b') w b
    in map (\layer -> act layer) res


  let add_padding (padding:i32) (X:[][]t) : [][]t =
    let width = length X + padding*2
    let height = length X[0] + padding *2
    let total_elem = width * height
    let index =  (flatten (map (\i -> (map (\j -> (i,j)) (0..<length X))) (0..<length X[0])))
    let offsets = map (\(i,j) -> padding*width + padding + width * i + j) index
    let retval = scatter (map (\_ -> R.(i32 0)) (0..<total_elem)) (offsets) (flatten X)
    in unflatten height width retval


  let backward (act:act) (l_layer:bool) ((w,b): weights) (input:input) (error:error_in) : gradients =
    -- let flip_filters = map (\x -> flip_matrix x) w
    if l_layer then
      (error, (w,b))
    else
      let res          = map2 (\w' b' -> convolve input w' b') w b
      let deriv        = map (\x -> act x) res
      let delta        = map2 (\x y -> util.mult_matrix x y ) error deriv
      let grad_b       = map (\x -> (R.sum) (map (\y -> (R.sum) y) x) ) delta
      let grad_w       = map (\e -> convolve input e R.(i32 0)) delta
      let delta_flip   = map (\x -> flip_matrix x) delta


      -- let flip_filters = map (\x -> flip_matrix x) w
      -- let add_pad      = map (\x -> add_padding 2 x) error
      -- let error'       = map (\w' -> convolve add_pad w' R.(i32 0)) flip_filters

    in (delta_flip, (grad_w, grad_b))


  let update (alpha:t) ((w,b):weights) ((wg,bg):weights) =
    let bg_scaled = map (\x -> R.(alpha * x)) bg
    let wg_scaled = map (\x -> util.scale_matrix x alpha ) wg
    let w'        = map2 (\w' x -> util.sub_matrix w' x ) w wg_scaled
    let b'        = map2 (\b' bg' -> R.(b' - bg')) b bg_scaled
     in (w', b')

  let layer  ((filters, kernel):input_params) (act:(act,act))   =
    let w:[][][]t  = random.gen_random_array_3d (kernel, kernel, filters) 1
    let b :[]t = map (\_ -> R.(i32 0)) (0..<filters)
   in
    (\w input -> (input, forward act.1 w input),
     (backward act.2),
      update,
     (w,b))

}
