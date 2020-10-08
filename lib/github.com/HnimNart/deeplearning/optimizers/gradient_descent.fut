import "optimizer_type"
import "../nn_types"
import "../util"

-- | Plain vanilla gradient descent optimizer
--   with mean gradient and constant learning rate
module gradient_descent (R:real) : optimizer_type
                                   with t = R.t
                                   with learning_rate = R.t = {

  type t = R.t
  type learning_rate = t

  module util = utility R

  let apply_grad_gd (alpha:learning_rate)
                    (batch_size:i64)
                    (m: i64) (n: i64)
                    ((w,b): (std_weights [m][n] [m] t))
                    ((wg,bg): (std_weights [m][n] [m] t)) =

    let wg_mean   = map (map R.((/i64 batch_size))) wg
    let bg_mean   = map (R.((/i64 batch_size))) bg

    let wg_scaled = util.scale_matrix wg_mean alpha
    let bg_scaled = util.scale_v bg_mean alpha

    let w'        = util.sub_matrix w wg_scaled
    let b'        = util.sub_v b bg_scaled

    in (w', b')

  let train [K] 'w 'g 'o 'e2 'i
            ({forward=f,
              backward=b,
              weights=w}:NN i w o g o e2 (apply_grad3 t))
            (alpha:learning_rate)
            (input:[K]i)
            (labels:[K]o)
            (batch_sz: i64)
            ({f=_, fd=loss'}:loss_func o t) =

    let i = 0
    let apply_g = apply_grad_gd alpha batch_sz
    let (w',_) = loop (w, i) while i < length input do
                   let input'          = take batch_sz (drop i input)
                   let label'          = take batch_sz (drop i labels)
                   let (cache, output) = f batch_sz true w input'
                   let error           = map2 (\o l -> loss' o l) output label'
                   let (_, w')         = b batch_sz false apply_g w cache error
                   in (w', i + batch_sz)
    in {forward = f, backward = b,  weights = w'}

}
