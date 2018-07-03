import "util"
import "activations"
import "/futlib/linalg"
import "classifications"

type forwards 'input 'w 'output 'garbage  = w -> input -> (garbage, output) --- NN
type backwards 'g 'w  'err1  'err2 = w -> g ->  err1 -> (err2, w)

-- type NN 'input 'w 'labels 'output 'delta 'new_w      = (forwards input w output, train input w  delta new_w output, w)
type NN 'input 'w 'output 'g 'e1 'e2  = (forwards input w output g,
                                                backwards g w e1 e2,
                                                w)

   ---- input weights labels output delta new_weights
type dense 't = NN ([][]t) ([][]t, [][]t) ([][]t) ([][]t) ([][]t)  ([][]t)

module random = normal_random_array f32
module activation = activation_funcs_coll f32
module lalg   = linalg f32
module cross = softmax_cross_entropy_with_logits f32
module util  = utility_funcs f32

let dense_forward [m][n][k] (act_id:i32) ((w,b):([m][n]f32, [m][1]f32)) (input:[n][k]f32) : ([m][k]f32) =
  let product = lalg.matmul w input
  let product' = map2 (\xr yr -> map2 (\x y -> (x + y)) xr yr) product b
  let act = activation.calc_activation (flatten product') act_id
  in unflatten m k act

let dense_backwards  (act_id:i32) (l_layer:bool) ((w,b):([][]f32, [][]f32)) (input:[][]f32) (error:[][]f32) =
  if l_layer then
let grad = lalg.matmul error (transpose input)

    let error_scaled = util.scaleMatrix error 0.01
    let grad_scaled = util.scaleMatrix grad 0.01
    let w'          = util.subMatrix w grad_scaled
    let b'          = util.subMatrix b error_scaled
    let error'      = lalg.matmul (transpose w) error
    in (error', (w', b'))
  else
    let res = lalg.matmul (w) input
    let (res_m, res_n) = (length res, length res[0])
    let deriv = unflatten res_m res_n (activation.calc_derivative (flatten res) act_id)
    let delta = util.multMatrix error deriv
    let grad  = lalg.matmul delta (transpose input)
    let delta_scaled = util.scaleMatrix delta 0.01
    let grad_scaled = util.scaleMatrix grad 0.01
    let w'       = util.subMatrix w grad_scaled
    let b'       = util.subMatrix b delta_scaled
    let error'   = lalg.matmul (transpose w) delta
    in (error', (w', b'))


let dense ((m,n):(i32,i32)) (act_id: i32) (l_layer:bool) : dense f32 =
  let w = unflatten n m (random.gen_random_array (m*n) )
  let b = unflatten n 1 (random.gen_random_array (n) )
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

let train 'w  'g 'e2  ((f,b,w):NN ([][]f32) w ([][]f32) g ([][]f32) e2) (input:[][]f32) (labels:[][]f32) (alpha:f32) =
  let (os, output) = f w (transpose input)
  let error = cross.derivative (transpose labels) output
  let (_, w') = b w os error
  in (f,b,w')




-- let accuracy 'w  'g 'e2 'wo ((f,_,w):NN ([][]f32) w ([][]f32) g ([][]f32) e2 wo) (input:[][]f32) (labels:[][]f32)  =
--   let (_, output) = f w (transpose input)
