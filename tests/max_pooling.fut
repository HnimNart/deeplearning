import "../lib/layers/max_pooling"

module max = max_pooling_2d f64
let max_layer = max.init (2,2) () 0

-- ==
-- entry: max_pooling_fwd
-- input{[[[[23.0,4.0,16.0,90.0],[12.0,32.0,12.0,45.0],
--          [5.0,7.0,8.0,9.0],[2.0,12.0,14.0,56.0]]]] }
-- output{ [[[[32.0, 90.0],[12.0,56.0]]]] }

entry max_pooling_fwd input =
    let (_, output) = max_layer.forward false max_layer.weights input
     in output

-- ==
-- entry: max_pooling_cache
-- input{[[[[23.0,4.0,16.0,90.0],
--          [12.0,32.0,12.0,45.0],
--          [5.0,7.0,8.0,9.0],
--          [2.0,12.0,14.0,56.0]]]] }
-- output{ [[[[5, 3],
--            [13, 15]]]]}

entry max_pooling_cache input =
    let (cache, _) = max_layer.forward true max_layer.weights input
     in cache

-- ==
-- entry: max_pooling_bwd
-- input {[[[[23.0,4.0,16.0,90.0],[12.0,32.0,12.0,45.0],
--          [5.0,7.0,8.0,9.0],[2.0,12.0,14.0,56.0]]]] }
-- output{[[[[0.0,0.0,0.0,90.0],[0.0,32.0,0.0,0.0],
--          [0.0,0.0,0.0,0.0],[0.0,12.0,0.0,56.0]]]] }

entry max_pooling_bwd input =
    let (c, output) = max_layer.forward true max_layer.weights input
    let (err, _)  = max_layer.backward false max_layer.weights c output
     in err
