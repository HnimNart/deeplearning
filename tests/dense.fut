import "../lib/github.com/HnimNart/deeplearning/deep_learning"
import "../lib/github.com/HnimNart/deeplearning/util"

module dl = deep_learning f64
module util = utility f64

let dense = dl.layers.dense 4 3 dl.nn.identity 1

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

let updater _ _ = (apply_grad_gd 0.1 1)



-- ==
-- entry: dense_fwd
-- input {[[1.0, 2.0, 3.0, 4.0],
--         [2.0, 3.0, 4.0, 5.0],
--         [3.0, 4.0, 5.0, 6.0]]
--
--        [[1.0,  2.0,  3.0,  4.0],
--         [5.0,  6.0,  7.0,  8.0],
--         [9.0, 10.0, 11.0, 12.0]]
--
--         [1.0, 2.0, 3.0]}
--
-- output {[[ 31.0,  72.0, 113.0],
--          [ 41.0,  98.0, 155.0],
--          [ 51.0, 124.0, 197.0]]}

entry dense_fwd [K] (input: [K][]f64) w b =
  let (_, output) = dense.forward K false (w,b) input
  in output

-- ==
-- entry: dense_cache_1
-- input {[[1.0, 2.0, 3.0, 4.0],
--         [2.0, 3.0, 4.0, 5.0],
--         [3.0, 4.0, 5.0, 6.0]]
--
--        [[1.0,  2.0,  3.0,  4.0],
--         [5.0,  6.0,  7.0,  8.0],
--         [9.0, 10.0, 11.0, 12.0]]
--
--         [1.0, 2.0, 3.0]}
--
-- output {[[1.0, 2.0, 3.0, 4.0],
--          [2.0, 3.0, 4.0, 5.0],
--          [3.0, 4.0, 5.0, 6.0]]}

entry dense_cache_1 [K] (input: [K][]f64) w b =
  let (cache, _) = dense.forward K true (w,b) input
  in map (.1) cache

-- == -- entry: dense_cache_2
-- input {[[1.0, 2.0, 3.0, 4.0],
--         [2.0, 3.0, 4.0, 5.0],
--         [3.0, 4.0, 5.0, 6.0]]
--
--        [[1.0,  2.0,  3.0,  4.0],
--         [5.0,  6.0,  7.0,  8.0],
--         [9.0, 10.0, 11.0, 12.0]]
--
--         [1.0, 2.0, 3.0]}
--
-- output {[[31.0,  72.0,  113.0],
--          [41.0,  98.0,  155.0],
--          [51.0, 124.0,  197.0]]}

entry dense_cache_2 [K] (input: [K][]f64) w b =
  let (cache, _) = dense.forward K true (w,b) input
  in map (.1) cache

-- ==
-- entry: dense_bwd_err
-- input {[[1.0, 2.0, 3.0, 4.0],
--         [2.0, 3.0, 4.0, 5.0],
--         [3.0, 4.0, 5.0, 6.0]]
--
--        [[1.0,  2.0,  3.0,  4.0],
--         [5.0,  6.0,  7.0,  8.0],
--         [9.0, 10.0, 11.0, 12.0]]
--
--         [1.0, 2.0, 3.0]}
--
-- output {[[1408.0, 1624.0, 1840.0, 2056.0],
--          [1926.0, 2220.0, 2514.0, 2808.0],
--          [2444.0, 2816.0, 3188.0, 3560.0]] }

entry dense_bwd_err [K] (input: [K][]f64) w b =
  let (cache, output) = dense.forward K true (w,b) input
  let (err, _) = dense.backward K false updater (w,b) cache output in err


-- ==
-- entry: dense_bwd_dW
-- input {[[1.0, 2.0, 3.0, 4.0],
--         [2.0, 3.0, 4.0, 5.0],
--         [3.0, 4.0, 5.0, 6.0]]
--
--        [[1.0,  2.0,  3.0,  4.0],
--         [5.0,  6.0,  7.0,  8.0],
--         [9.0, 10.0, 11.0, 12.0]]
--
--         [1.0, 2.0, 3.0]}
--
-- output {[[-25.60,  -36.90,  -48.20,  -59.50],
--          [-59.00,  -87.40, -115.80, -144.20],
--          [-92.40, -137.90, -183.40, -228.90]]}


entry dense_bwd_dW [K] (input: [K][]f64) w b =
  let (cache, output) = dense.forward K true (w,b) input
  let (_, (w',_)) = dense.backward K false updater (w,b) cache output
  in w'

-- ==
-- entry: dense_bwd_dB
-- input {[[1.0, 2.0, 3.0, 4.0],
--         [2.0, 3.0, 4.0, 5.0],
--         [3.0, 4.0, 5.0, 6.0]]
--
--        [[1.0,  2.0,  3.0,  4.0],
--         [5.0,  6.0,  7.0,  8.0],
--         [9.0, 10.0, 11.0, 12.0]]
--
--         [1.0, 2.0, 3.0]}
--
-- output {[-11.30, -27.40, -43.50]}

entry dense_bwd_dB [K] (input: [K][]f64) w b =
  let (cache, output) = dense.forward K true (w,b) input
  let (_, (_,b')) = dense.backward K false updater (w,b) cache output
  in b'
