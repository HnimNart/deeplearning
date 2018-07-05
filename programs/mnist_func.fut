import "../lib3/network"

let l1    = dense (784, 256) 0 false
let l2    = dense (256, 128) 0 false
let l3    = dense (128, 10) 0 true
let nn'   = combine l1 l2
let model = combine nn' l3


let main [m][n][d] (input: [m][d]f32) (labels: [m][n]f32) =
  let nn = train model (transpose input) (transpose labels) 0.01
  in (accuracy nn (transpose input) (transpose labels),
      accuracy model (transpose input) (transpose labels))
