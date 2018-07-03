import "../lib3/network"

let n = 1
let n_input = 728
let n_class = 2
let input1  = unflatten n n_input (random.gen_random_array (n_input*n))
-- let labels = unflatten n n_class (random.gen_random_array (n_class*n))
-- let w = unflatten 256 784 (random.gen_random_array (784*256) 1)
-- let b = unflatten 256 1 (random.gen_random_array 256 1)


let l1    = dense (784, 256) 0 false
let l2    = dense (256, 128) 0 false
let l3    = dense (128, 10) 0 true
let nn'   = combine l1 l2
let model = combine nn' l3


let main [m][n][d] (input: [m][d]f32) (labels: [m][n]f32) =
  -- (length (transpose input1)[0], length (transpose [input[0]])[0] )
  let data_sets = 64000
  let batch_size = 1
  let i = 0
  let m  = map (\i -> let (_,_,v) = train model [input[i]] [labels[i]] 0.01 in v) (0..<1000) in m[999]
