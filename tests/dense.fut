import "../lib/deep_learning"
module dl = deep_learning f64

let dense = dl.layers.dense (4,3) dl.nn.identity 1

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

entry dense_fwd input w b =
  let (_, output) = dense.forward false (w,b) input
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

entry dense_cache_1 input w b =
  let (cache, _) = dense.forward true (w,b) input
  in cache.1

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

entry dense_cache_2 input w b =
  let (cache, _) = dense.forward true (w,b) input
  in cache.1


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

entry dense_bwd_err input w b =
  let (cache, output) = dense.forward true (w,b) input
  let (err, _) = dense.backward false (w,b) cache output in err


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
-- output {[[ 266.00,  389.00,  512.00,  635.00],
--          [ 640.00,  934.00, 1228.00, 1522.00],
--          [1014.00, 1479.00, 1944.00, 2409.00]]}

entry dense_bwd_dW input w b =
  let (cache, output) = dense.forward true (w,b) input
  let (_, (w',_)) = dense.backward false (w,b) cache output
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
-- output {[123.000000f64, 294.000000f64, 465.000000f64]}

entry dense_bwd_dB input w b =
  let (cache, output) = dense.forward true (w,b) input
  let (_, (_,b')) = dense.backward false (w,b) cache output
  in b'
