import "optimizer_types"
import "../nn_types"
import "../util"

module GradientDescent (R:real) : trainer with t = R.t
                                          with updater = updater ([][]R.t, []R.t) = {

  type t = R.t
  type updater = updater ([][]t, []t)

  module util = utility R

  let update_weights (alpha:t) (batch_size:i32) ((w,b):([][]t, []t)) ((wg,bg):([][]t, []t)) =


      let wg_mean   = map (map R.((/i32 (batch_size)))) wg
      let bg_mean   = map (R.((/i32 (batch_size)))) bg
      -- let wg' = map (\xr -> map (\x -> R.(if x > i32 10 then i32 10 else if x < (negate (i32 10)) then (negate (i32 10)) else x)) xr) wg_mean

      let wg_scaled = util.scale_matrix wg_mean alpha
      let bg_scaled = util.scale_v bg_mean alpha

      let w'        = util.sub_matrix w wg_scaled
      let b'        = util.sub_v b bg_scaled
      in (w', b')


  let train 'w 'g 'o 'e2 'i ((f,b,u,w):NN ([]i) w ([]o) g ([]o) e2 updater) (alpha:t)
                          (input:[]i) (labels:[]o) (step_sz: i32) (loss:(o -> o -> t , o -> o -> o)) =


    let i = 0
    let (w',_) = loop (w, i) while i < length input - 1 do
             let inp' = input[i:i+step_sz]
             let lab  = labels[i:i+step_sz]
             let (os, output) = f true w (inp')
             let error = map2 (\o l -> loss.2 o l) output lab
             let (_, grads) = b w os error
             let w'   = u (update_weights alpha step_sz) w grads
             in (w', i+ step_sz)
    in (f,b, u,w')

}