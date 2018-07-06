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



-- module max_pooling_2d (R:real) : layer =  {


--   type t = R.t
--   type input = [][][][]t
--   type weights = ()
--   type output = [][][][]t
--   type garbage = [][][][]t
--   type error_in = [][][][]t
--   type error_out = [][][][]t
--   type gradients = (error, weights)
--   type input_params = (i32, i32)

--   type act = ()
--   type layer = NN input weights output garbage error_in error_out


-- }

module conv2d (R:real) : layer with t = R.t
                               with input = [][][][]R.t
                               with input_params = (i32,i32)
                               with weights = ([][][]R.t, []R.t)
                               with output  = ([][][][]R.t)
                               with error_in = ([][][][]R.t)

                               with error_out = ([][][][]R.t)
                               with gradients = ([][][][]R.t ,([][][]R.t, []R.t))
                               with layer = NN ([][][][]R.t) ([][][]R.t,[]R.t) ([][][][]R.t) ([][][][]R.t) ([][][][]R.t) ([][][][]R.t) (R.t)

                               with act = ([][]R.t -> [][]R.t) = {


  type t = R.t
  type input = [][][][]t
  type weights  = ([][][]t, []t)
  type output = [][][][]t
  type garbage  = [][][][]t
  type error_in = [][][][]t
  type error_out = [][][][]t
  type gradients = (error_out, weights)
  type input_params = (i32 ,i32)

  type act = [][]t -> [][]t
  type layer = NN input weights output garbage error_in error_out t

  module lalg   = linalg R
  module util   = utility_funcs R
  module random = normal_random_array R


  let red_matrix (input: [][]t) (w:[][]t) (b:t) =
    R.((sum (flatten (util.multMatrix input w))) + b)


  let calc (input: [][][]t) (w: [][]t) (b:t) =
    let (m,n) = (length input[0], length input[0,0])
    let (wm, wn) = (length w, length w[0])
    let (m_d, n_d) = (m-wm+1, n -wn+1)
    let res = map (\layer -> map (\i -> map (\j -> red_matrix layer[i:i+wm,j:j+wn] w b) (0..<n_d)) (0..<m_d)) input
    let ne = replicate (m_d) (replicate (n_d) R.(i32 0))
    in reduce (util.add_matrix) ne res

  let forward (act:act) ((w,b):weights) (input:input) : output =
    let res = map (\img -> map2 (\w' b' -> calc img w' b') w b) input
    in map (\img -> map (\layer -> act layer ) img) res

  let backward (act:act) (l_layer:bool) ((w,b): weights) (input:input) (error:error_in) : gradients =
    if l_layer then
    (error, (w,b))
    else
    (error, (w,b))

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

-- module con = conv2d f32
-- module rand = normal_random_array f32
-- module activations = activations f32

-- let input  = map (\_ -> map (\_ -> rand.gen_random_array_2d (7,7) 1) (0..<100)) (0..<3)
-- -- let w = map (\_ -> rand.gen_random_array_2d (3,3) 1) (0..<1)
-- let l = con.layer 3 3

-- let main  =
--   let res = (con.forward activations.Identity_2d.1 l input) in res
