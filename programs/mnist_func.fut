import "../lib3/network"

let n = 10
let n_input = 784
let n_class = 10
let input  = unflatten n_input n (random.gen_random_array (n_input*n))
let labels = unflatten n_class n (random.gen_random_array (n_class*n))
-- let w = unflatten 256 784 (random.gen_random_array (784*256) 1)
-- let b = unflatten 256 1 (random.gen_random_array 256 1)


let l1    = dense (784, 256) 0 false
let l2    = dense (256, 128) 0 false
let l3    = dense (128, 10) 0 true
let nn'   = combine l1 l2
let model = combine nn' l3


-- let main [m][n][d] (input: [m][d]f32) (labels: [m][n]f32) =
let main =
  -- (length (transpose input1)[0], length (transpose [input[0]])[0] )
  let data_sets = 64000
  let (f, _, w) = model
  in f w input
  -- let tmp = train model input labels 0.01 in tmp.3
