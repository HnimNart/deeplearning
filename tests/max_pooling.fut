import "../lib/github.com/HnimNart/deeplearning/layers/max_pooling"
import "../lib/github.com/HnimNart/deeplearning/util"

module max = max_pooling_2d f64
module util = utility f64

let max_layer = max.init (2,2) () 0


let apply_grad_gd (alpha:f64)
                  (batch_size:i32)
                  ((w,b):([][]f64, []f64))
                  ((wg,bg):([][]f64,[]f64)) =

  let wg_mean   = map (map f64.((/i32 batch_size))) wg
  let bg_mean   = map (f64.((/i32 batch_size))) bg

  let wg_scaled = util.scale_matrix wg_mean alpha
  let bg_scaled = util.scale_v bg_mean alpha

  let w'        = util.sub_matrix w wg_scaled
  let b'        = util.sub_v b bg_scaled

  in (w', b')


-- ==
-- entry: max_pooling_fwd
-- input{[[[[23.0, 4.0,  16.0,  90.0],
--          [12.0, 32.0, 12.0,  45.0],
--          [5.0,  7.0,  8.0,   9.0],
--          [2.0,  12.0, 14.0,  56.0]]]] }
-- output{ [[[[32.0, 90.0],
--            [12.0, 56.0]]]] }

entry max_pooling_fwd input =
    let (_, output) = max_layer.forward false max_layer.weights input
     in output

-- ==
-- entry: max_pooling_cache
-- input{[[[[23.0, 4.0,  16.0,  90.0],
--          [12.0, 32.0, 12.0,  45.0],
--          [5.0,  7.0,  8.0,   9.0],
--          [2.0,  12.0, 14.0,  56.0]]]] }
-- output{ [[[[5i32,   3i32],
--            [13i32, 15i32]]]]}

entry max_pooling_cache input =
    let (cache, _) = max_layer.forward true max_layer.weights input
     in cache

-- ==
-- entry: max_pooling_bwd
-- input{[[[[23.0, 4.0,  16.0,  90.0],
--          [12.0, 32.0, 12.0,  45.0],
--          [5.0,  7.0,  8.0,   9.0],
--          [2.0,  12.0, 14.0,  56.0]]]] }
-- output{[[[[0.0, 0.0,  0.0, 90.0],
--           [0.0, 32.0, 0.0,  0.0],
--           [0.0, 0.0,  0.0,  0.0],
--           [0.0, 12.0, 0.0, 56.0]]]] }

entry max_pooling_bwd input =
    let (c, output) = max_layer.forward true max_layer.weights input
    let (err, _)  = max_layer.backward false (apply_grad_gd f64.(0.1) 1) max_layer.weights c output
     in err
