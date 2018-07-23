import "optimizer_types"
import "../nn_types"
import "../util"



-- | Plain vanilla gradient descent optimizer
--   with mean gradient
module GradientDescent (R:real) : trainer with t = R.t
                                          with alpha = R.t = {

  type t = R.t
  type alpha = t
  module util = utility R

  let apply_weights (alpha:alpha) (batch_size:i32) ((w,b):(std_weights t)) ((wg,bg):(std_weights t)) =

      let wg_mean   = map (map R.((/i32 batch_size))) wg
      let bg_mean   = map (R.((/i32 batch_size))) bg

      let wg_scaled = util.scale_matrix wg_mean alpha
      let bg_scaled = util.scale_v bg_mean alpha

      let w'        = util.sub_matrix w wg_scaled
      let b'        = util.sub_v b bg_scaled
    in (w', b')

  let train [n] 'w 'g 'o 'e2 'i (nn:NN ([]i) w ([]o) g ([]o) e2 (apply_grad t))
                             (alpha:alpha)
                             (input:[n]i)
                             (labels:[n]o)
                             (batch_sz: i32)
                             (loss:(o -> o -> t , o -> o -> o)) =

    let i = 0
    let w = nn.weights
    let (w',_) = loop (w, i) while i < length input do
                   let inp'            = input[i:i+batch_sz]
                   let lab             = labels[i:i+batch_sz]
                   let (cache, output) = nn.forward true nn.weights (inp')
                   let error           = map2 (\o l -> loss.2 o l) output lab
                   let (_, grads)      = nn.backward false nn.weights cache error
                   let w'              = nn.update (apply_weights alpha batch_sz) w grads
                   in (w', i + batch_sz)
    in {forward = nn.forward,backward = nn.backward, update = nn.update , weights = w'}

}