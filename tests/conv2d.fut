import "../lib/github.com/HnimNart/deeplearning/deep_learning"
import "../lib/github.com/HnimNart/deeplearning/util"

module dl = deep_learning f64
module util = utility f64

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

let updater _ _ = apply_grad_gd 0.1 1

let act (d: i32) : activation_func ([d]f64) =
  dl.nn.identity : activation_func ([d]f64)

let conv = dl.layers.conv2d 2 3 3
                            2 1 2
                            2 2 act 1

-- ==
-- entry: conv2d_fwd
-- input { [[[[1.0,   2.0,  3.0],
--            [4.0,   5.0,  6.0],
--            [7.0,   8.0,  9.0]],
--           [[10.0, 11.0, 12.0],
--            [13.0, 14.0, 15.0],
--            [16.0, 17.0, 18.0]]]]
--
--         [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
--          [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]
--
--         [1.0,2.0] }
-- output {[[[[357.0, 393.0],
--            [465.0, 501.0]],
--           [[186.0, 222.0],
--            [294.0, 330.0]]]]}

entry conv2d_fwd [K] (data: [K][][][]f64) w b  =
  let (_, output) = conv.forward K false (map (unflatten_3d 2 2 2) w,b) data in output

-- ==
-- entry: conv2d_cache_bias
-- input { [[[[1.0,   2.0,  3.0],
--            [4.0,   5.0,  6.0],
--            [7.0,   8.0,  9.0]],
--           [[10.0, 11.0, 12.0],
--            [13.0, 14.0, 15.0],
--            [16.0, 17.0, 18.0]]]]
--
--         [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
--          [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]
--
--         [1.0,2.0] }
--
-- output {[[[[357.0, 393.0], [465.0, 501.0]],
--            [[186.0, 222.0], [294.0, 330.0]]]]}

entry conv2d_cache_bias [K] (data: [K][][][]f64) w b  =
  let (cache, _) = conv.forward K true (map (unflatten_3d 2 2 2) w,b) data in map (.2) cache

-- ==
-- entry: conv2d_cache_matrix
-- input { [[[[1.0,   2.0,  3.0],
--            [4.0,   5.0,  6.0],
--            [7.0,   8.0,  9.0]],
--           [[10.0, 11.0, 12.0],
--            [13.0, 14.0, 15.0],
--            [16.0, 17.0, 18.0]]]]
--
--         [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
--          [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]
--
--         [1.0,2.0] }
--
-- output {[[[[[[1.0,   2.0], [4.0,  5.0]],
--             [[2.0,   3.0], [5.0,  6.0]]],
--            [[[4.0,   5.0], [7.0,  8.0]],
--             [[5.0,   6.0], [8.0,  9.0]]]],
--           [[[[10.0, 11.0], [13.0, 14.0]],
--             [[11.0, 12.0], [14.0, 15.0]]],
--            [[[13.0, 14.0], [16.0, 17.0]],
--             [[14.0, 15.0], [17.0, 18.0]]]]]]
--        }

entry conv2d_cache_matrix [K] (data: [K][][][]f64) w b  =
  let (cache, _) = conv.forward K true (map (unflatten_3d 2 2 2) w,b) data
  in map (.1) cache

-- ==
-- entry: conv2d_bwd_error
-- input { [[[[1.0,   2.0,  3.0],
--            [4.0,   5.0,  6.0],
--            [7.0,   8.0,  9.0]],
--           [[10.0, 11.0, 12.0],
--            [13.0, 14.0, 15.0],
--            [16.0, 17.0, 18.0]]]]
--
--         [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
--          [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]
--
--         [1.0,2.0] }
--
-- output { [[[[1845.0,  4185.0,  2340.0],
--             [5004.0, 10998.0,  5994.0],
--             [3159.0,  6813.0,  3654.0]],
--            [[2529.0,  5553.0,  3024.0],
--             [6372.0, 13734.0,  7362.0],
--             [3843.0,  8181.0,  4338.0]]]]}

entry conv2d_bwd_error [K] (data: [K][][][]f64) w b  =
  let (cache, output) = conv.forward K true (map (unflatten_3d 2 2 2) w,b) data
  let (err, _) = conv.backward K false updater (map (unflatten_3d 2 2 2) w,b) cache output in err

-- ==
-- entry: conv2d_bwd_dW
-- input { [[[[1.0,   2.0,  3.0],
--            [4.0,   5.0,  6.0],
--            [7.0,   8.0,  9.0]],
--           [[10.0, 11.0, 12.0],
--            [13.0, 14.0, 15.0],
--            [16.0, 17.0, 18.0]]]]
--
--         [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
--          [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]
--
--         [1.0,2.0] }
--
-- output {[[[[-549.80, -720.40], [-1062.60, -1233.20]], [[-2090.20, -2260.80], [-2603.00, -2773.60]]],
--          [[[-337.60, -441.80],  [-649.20,  -753.40]], [[-1270.40, -1374.60], [-1582.00, -1686.20]]]]}

entry conv2d_bwd_dW [K] (data: [K][][][]f64) w b =
  let (cache, output) = conv.forward K true (map (unflatten_3d 2 2 2) w,b) data
  let (_, (w',_)) = conv.backward K false updater (map (unflatten_3d 2 2 2) w,b) cache output
  in w'

-- ==
-- entry: conv2d_bwd_dB
-- input { [[[[1.0,   2.0,  3.0],
--            [4.0,   5.0,  6.0],
--            [7.0,   8.0,  9.0]],
--           [[10.0, 11.0, 12.0],
--            [13.0, 14.0, 15.0],
--            [16.0, 17.0, 18.0]]]]
--
--         [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
--          [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]
--
--         [1.0,2.0] }
--
-- output {[-170.60, -101.20]}


entry conv2d_bwd_dB [K] (data: [K][][][]f64) w b  =
  let (cache, output) = conv.forward K true (map (unflatten_3d 2 2 2) w,b) data
  let (_, (_,b')) = conv.backward K false updater (map (unflatten_3d 2 2 2) w,b) cache output
  in b'
