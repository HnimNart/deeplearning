import "../lib/layers/conv2d"
import "../lib/activations"

module conv2d = conv2d f32
module act    = activations f32

let conv2dlayer = conv2d.layer (1, 2, 1) act.Identity_1d

let w :[][]f32      = [[1,2,3,4]]
let b:[]f32         = [1]
let input:[][]f32 = [[1,2,3,4],[5,6,7,8], [9,10,11,12], [13,14,15,16]]
let input' = [input, input]
let input4d = [input', map (\x -> reverse x) input']


let get_dims (X:[][][][]f32) =
 (length X, length X[0], length X[0,0], length X[0,0,0])

let main =
  let (f, cb, _, _) = conv2dlayer
  let (_, os) =f (w,b) input4d
  let (e, (w',b')) = cb false (w,b) input4d os in e


--[[[45.000000f32, 55.000000f32, 65.000000f32], [85.000000f32, 95.000000f32, 105.000000f32], [125.000000f32, 135.000000f32, 145.000000f32]]]
