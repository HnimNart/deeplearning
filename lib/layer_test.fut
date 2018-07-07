import "types"
import "activations"
import "/futlib/linalg"
import "util"


module type layer = {

  type t
  type input
  type weights
  type output
  type error_in
  type error_out
  type gradients

  type act
  type layer

  type input_params

  val layer:  input_params -> (act,act) -> layer
  -- val forward: act -> weights -> input -> output
  -- val backward:  act -> bool ->  weights ->  input -> error_in -> gradients
  -- val layer: (i32, i32) -> (act, act) -> layer

  -- val get_ws: layer -> weights
  -- val get_f: layer -> weights -> input -> (input, output)
  -- val get_b: layer -> bool -> weights ->  input -> error_in -> gradients
}


-- module reshape (R:real ): layer = {
  -- let forwards = (_:act) (_:weights) (input:input) : output =
    -- map (\xr -> flatten (map (\x -> flatten)) xr) input

-- }

-- module max_pooling_2d (R:real) : layer with t = R.t
--                                        with input = [][][][]R.t
--                                        with input_params = (i32,i32)
--                                        with weights = ()
--                                        with output  = ([][]R.t)
--                                        with error_in = ([][]R.t)
--                                        with error_out = ([][][][]R.t)
--                                        with gradients = ([][][][]R.t,())
--                                        with layer = NN ([][][][]R.t) () ([][][][]R.t) ([][][][]R.t) ([][]R.t) ([][][][]R.t) ()
--                                with act = ()  =  {


--   type t = R.t
--   type input = [][][][]t
--   type weights = ()
--   type output = [][][][]t
--   type garbage = [][][][]t
--   type error_in = [][][][]t
--   type error_out = [][][][]t
--   type gradients = (error_out, weights)
--   type input_params = (i32, i32)

--   type act = ()
--   type layer = NN input weights output garbage error_in error_out


--   -- let layer ((m,n):(i32, i32)) (():act) =


-- }

module conv2d (R:real) : layer with t = R.t
                               with input = [][][]R.t
                               with input_params = (i32,i32)
                               with weights = ([][][]R.t, []R.t)
                               with output  = ([][][]R.t)
                               with error_in = ([][][]R.t)
                               with error_out = ([][][]R.t)
                               with gradients = ([][][]R.t ,([][][]R.t, []R.t))
                               with layer =
NN ([][][]R.t) ([][][]R.t,[]R.t) ([][][]R.t) ([][][]R.t) ([][][]R.t) ([][][]R.t) (R.t)

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
  module util   = utility_funcs R
  module random = normal_random_array R


  let flip_matrix (X:[][]t) =
    reverse (map (\x -> reverse x) X)

  let add_3d_matrix (X:[][][]t) (Y:[][][]t) =
    map2 (\x y -> map2 (\xr yr -> map2 (\x' y' -> R.(x' +  y')) xr yr) x y) X Y

  let add_2d_matrix (X:[][]t) (Y:[][]t) =
    map2 (\xr yr -> map2 (\x y -> R.(x +  y)) xr yr) X Y


  let red_matrix (input: [][]t) (w:[][]t) (b:t) =
    R.((sum (flatten (util.multMatrix input w))) + b)

  let convolve (input: [][][]t) (w: [][]t) (b:t) =
    let (m,n) = (length input[0], length input[0,0])
    let (wm, wn) = (length w, length w[0])
    let (m_d, n_d) = (m-wm+1, n -wn+1)
    let res =
      map (\layer -> map (\i -> map (\j -> red_matrix layer[i:i+wm,j:j+wn] w b) (0..<n_d)) (0..<m_d)) input
    let ne = replicate (m_d) (replicate (n_d) R.(i32 0))
    in reduce (util.add_matrix) ne res

  let forward (act:act) ((w,b):weights) (input:input) : output =
    let res = map2 (\w' b' -> convolve input w' b') w b
    in map (\layer -> act layer) res


  let in_bounds = true
  let full_convolve (input:[][]t) (w:[][]t) = 0


  let backward (act:act) (l_layer:bool) ((w,b): weights) (input:input) (error:error_in) : gradients =
    -- let flip_filters = map (\x -> flip_matrix x) w
    if l_layer then
      (error, (w,b))
    else
      let res = map2 (\w' b' -> convolve input w' b') w b
      let deriv = map (\x -> act x) res
      let delta = map2 (\x y -> util.multMatrix x y) error deriv

    in (delta, (w,b))


  let update (alpha:t) ((w,b):weights) ((wg,bg):weights) =
    let bg_scaled = map (\x -> R.(alpha * x)) bg
    let wg_scaled = map (\x -> util.scaleMatrix x alpha ) wg
    let w'        = map2 (\w' x -> util.subMatrix w' x ) w wg_scaled
    let b'        = map2 (\b' bg' -> R.(b' - bg')) b bg_scaled
     in (w', b')

  let layer  ((filters, kernel):input_params) (act:(act,act))   =
    let w:[][][]t  = random.gen_random_array_3d (kernel, kernel, filters) 1
    let b :[]t = random.gen_random_array filters 1
      --map (\_ -> R.(i32 0)) (0..<filters)
   in
    (\w input -> (input, forward act.1 w input),
     (backward act.2),
      update,
     (w,b))

}

module con = conv2d f32
module rand = normal_random_array f32
module act = activations f32
module util = utility_funcs f32

let input  = map (\i -> rand.gen_random_array_2d (3,3) i) (0..<1)
let labels :[][]f32 = [[1.0, 0,0,0]]
let l = con.layer (1, 2) act.Identity_2d

let main  = let (f, b, _, w) = l
            let (g, out ) = f w input
            let out' = [flatten (flatten out)]
            let loss = map2 (\xr yr -> map2 (\x y -> y - x) xr yr) labels out'
            let error = map (\x -> unflatten 2 2 x) loss
            let (err, (w', b')) = b false w g error in (err)

            -- let (err, (w', b')) = b true w g loss in (w', err)

-- let w = [[4,5,6],[1,2,3]] in w[::-1]