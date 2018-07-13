import "../lib/tensorflow"
import "/futlib/linalg"
module tf = tensorflow f32
module lalg = linalg f32

let conv = tf.layers.Conv2d (2, 2, 1, 2 ) tf.nn.identity 1

let input1:[][][]f32 = [[[1,2,3],[4,5,6],[7,8,9]], [[10,11,12],[13,14,15],[16,17,18]]]
let input2:[][][]f32 = [[[10,11,12],[13,14,15],[16,17,18]], [[10,11,12],[13,14,15],[16,17,18]]]

let w:[][]f32 = [[1,2,3,4, 5,6,7,8],[8,7,6,5,4,3,2,1]]

let w' = w --[concat w (concat w w )]

let b:[]f32 = [0,0]

let im2col (x:[][][]f32) ((w_m, w_n):(i32, i32)) (idx:[](i32, i32)) : [][]f32 =
  unsafe transpose (map (\(i,j) ->  flatten (map (\layer -> flatten layer[i:i+w_m, j:j+w_n]) x)) idx)

let calc_index (stride:i32) ((m,n):(i32, i32)) =
    let row_index = map (\i -> i * stride) (0..<m)
    let col_index = map (\i -> i * stride) (0..<n)
    in flatten (map (\i -> map (\j -> (i,j) ) row_index) col_index)

let get_dims (X:[][][][]f32)  =
  (length X, length X[0], length X[0,0], length X[0,0,0])

let flip_matrix (X:[][]f32) =
  reverse (map (\x -> reverse  x) X)

let main =
  let (f, bf, _, _) =  conv
  let (os, output) = f true (w', b) [input1, input2]
  let (err, (w1,b')) = bf (w',b) os output in w1

  -- let delta = output
  -- let tmp =  map (\img -> map (\layer -> f32.sum (flatten layer) ) img) delta
  -- in map (f32.sum) (transpose tmp)
--   let delta_flipped = map (\x -> map (\x' -> flatten ( flip_matrix x') ) x ) delta
--   let input2 = os.2 --in delta[0]
--   -- in (length (input2[0]), length input2[0,0])
--   let w_grads =  map2 (\d i -> transpose (lalg.matmul i (transpose d))) delta_flipped input2
--  -- in w_grads
-- in    map (\d -> map (f32.sum ) (transpose d)) (transpose w_grads)
-- -- in  map (\img -> util.add_2d_matrix )

 -- [[[4776.000000f32, 6488.000000f32, 9912.000000f32, 11624.000000f32, 20184.000000f32, 21896.000000f32, 25320.000000f32, 27032.000000f32],
 --   [2712.000000f32, 3736.000000f32, 5784.000000f32, 6808.000000f32, 11928.000000f32, 12952.000000f32, 15000.000000f32, 16024.000000f32]],

 --  [[24504.000000f32, 26576.000000f32, 30720.000000f32, 32792.000000f32, 24504.000000f32, 26576.000000f32, 30720.000000f32, 32792.000000f32],
 --   [23160.000000f32, 25120.000000f32, 29040.000000f32, 31000.000000f32, 23160.000000f32, 25120.000000f32, 29040.000000f32, 31000.000000f32]]]
