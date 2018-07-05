import "util"
import "activations"
import "/futlib/linalg"
import "classifications"

type forwards 'input 'w 'output 'garbage  = w -> input -> (garbage, output) --- NN
type backwards 'g 'w  'err1  'err2 = w -> g ->  err1 -> (err2, w)

type NN 'input 'w 'output 'g 'e1 'e2  = (forwards input w output g,
                                         backwards g w e1 e2,
                                         w)

---- input weights labels output delta
type dense 't = NN ([][]t) ([][]t, [][]t) ([][]t) ([][]t) ([][]t)  ([][]t)

module random = normal_random_array f32
module activation = activation_funcs_coll f32
module lalg   = linalg f32
module cross = softmax_cross_entropy_with_logits f32
module util  = utility_funcs f32
module soft = softmax_stable f32

---- Each input is in a column
let dense_forward [m][n][k] (act_id:i32) ((w,b):([m][n]f32, [m][1]f32)) (input:[n][k]f32) : ([m][k]f32) =
  let product = lalg.matmul w input
  let product' = map2 (\xr b -> map (\x -> (x + b[0])) xr) product b
  let act = activation.calc_activation (flatten product') act_id
  in unflatten m k act


let dense_backwards [m][n][d]  (act_id:i32) (l_layer:bool) ((w,b):([][]f32, [][]f32)) (input:[d][n]f32) (error:[m][n]f32) =
  if l_layer then
    let error_corrected = (map (map ((/f32.(i32 n)))) error)
    let grad           = lalg.matmul error_corrected (transpose input)
    let error_reduced  = transpose  [map (f32.sum) error_corrected]
    let error_scaled   = util.scaleMatrix error_reduced 0.01
    let grad_scaled    = util.scaleMatrix grad 0.01
    let w'             = util.subMatrix w grad_scaled
    let b'             = util.subMatrix b error_scaled
    let error'         = lalg.matmul (transpose w) error_corrected
    in (error', (w', b'))
  else
    let res            = lalg.matmul (w) input
    let (res_m, res_n) = (length res, length res[0])
    let deriv          = unflatten res_m res_n (activation.calc_derivative (flatten res) act_id)
    -- let error_corrected = (map (map ((/f32.(i32 n)))) error)
    let delta          = util.multMatrix error deriv
    let grad           = lalg.matmul delta (transpose input)
    let delta_scaled   = util.scaleMatrix delta 0.01
    let grad_scaled    = util.scaleMatrix grad 0.01
    let b_grad         = transpose [map (f32.sum) delta_scaled]
    let w'             = util.subMatrix w grad_scaled
    let b'             = util.subMatrix b b_grad
    let error'         = lalg.matmul (transpose w) delta
    in (error', (w', b'))


let dense ((m,n):(i32,i32)) (act_id: i32) (l_layer:bool) : dense f32 =
  let w = unflatten n m (random.gen_random_array (m*n) 2)
  let b = unflatten n 1 (random.gen_random_array (n)  2)
  in (\w input -> (input, dense_forward act_id w input),
     (\w input error -> dense_backwards act_id l_layer w input error),
     (w,b))

let combine 'w1 'w2 'i1 'o1 'o2 'g1 'g2 'e1 'e2 'e22  ((f1, b1,ws1): NN i1 w1 o1 g1 e22 e1) ((f2, b2,ws2):NN o1 w2 o2 g2 e2 e22)
                                                                                  : NN i1 (w1,w2) (o2) (g1,g2) (e2) (e1) =
  ((\(w1, w2) (input) ->  let (g1, res)  = f1 w1 input
                          let (g2, res2) = f2 w2 res
                          in ((g1, g2), res2)),
   (\(w1,w2) (g1,g2) (error) ->
                          let (err2, w2') = b2  w2 g2 error
                          let (err1, w1') = b1 w1 g1 err2
                          in (err1, (w1', w2'))),
   (ws1, ws2))


let getCols [m][n] (M:[m][n]f32) (i:i32) (c:i32) : [m][c]f32 =
  M[:,i:i+c]

let train 'w  'g 'e2  ((f,b,w):NN ([][]f32) w ([][]f32) g ([][]f32) e2) (input:[][]f32) (labels:[][]f32) (alpha:f32) =
  let i = 0
  let batch_size = 128
  let (w',_) = loop (w, i) while i < 64000 do
           let inp' = getCols input i batch_size
           let lab  = getCols labels i batch_size
           let (os, output) = f w (inp')
           let error = cross.derivative (lab) output
           let (_, w') = b w os error
           in (w', i+ batch_size)
  in (f,b,w')


let predict 'w 'g 'e2 ((f,_,w):NN ([][]f32) w ([][]f32) g ([][]f32) e2) (input:[][]f32)  =
  let (_, output) = f w input
  in soft.classify2 output

let is_equal [n] (logits:[n]f32) (labels:[n]f32) =
  let logits_i:i32 = (reduce (\n i -> if unsafe (f32.(logits[n] > logits[i])) then n else i) 0 (iota n))
  let labels_i:i32 = (reduce (\n i -> if unsafe (f32.(labels[n] > labels[i])) then n else i) 0 (iota n))
  in if logits_i == labels_i  then 1 else 0

let accuracy [d][c] 'w 'g 'e2 (nn:NN ([][]f32) w ([][]f32) g ([][]f32) e2) (input:[][d]f32) (labels:[c][d]f32) : f32=
  let predictions = transpose (predict nn input)
  let labels_T    =   transpose (labels)
  let hits        = map2 (\x y -> is_equal x y) predictions labels_T
  let total       = reduce (+) 0 hits
  in f32.(i32 total / i32 d)
