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
                    (batch_size:i32)
                    ((w,b):(std_weights t))
                    ((wg,bg):(std_weights t)) =

    let wg_mean   = map (map R.((/i32 batch_size))) wg
    let bg_mean   = map (R.((/i32 batch_size))) bg

    let wg_scaled = util.scale_matrix wg_mean alpha
    let bg_scaled = util.scale_v bg_mean alpha

    let w'        = util.sub_matrix w wg_scaled
    let b'        = util.sub_v b bg_scaled

    in (w', b')

  let train [n] 'w 'g 'o 'e2 'i ({forward=f,
                                  backward=b,
                                  weights=w}:NN ([]i) w ([]o) g ([]o) e2 (apply_grad t))
                                (alpha:learning_rate)
                                (input:[n]i)
                                (labels:[n]o)
                                (batch_sz: i32)
                                ({f=_, fd=loss'}:loss_func o t) =

    let i = 0
    let applier = apply_grad_gd alpha batch_sz
    let (w',_) = loop (w, i) while i < length input do
                   let input'          = input[i:i+batch_sz]
                   let label'          = labels[i:i+batch_sz]
                   let (cache, output) = f true w (input')
                   let error           = map2 (\o l -> loss' o l) output label'
                   let (_, w')         = b false applier w cache error
                   in (w', i + batch_sz)
    in {forward = f, backward = b,  weights = w'}

}