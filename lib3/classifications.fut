

module type classification = {

  type t
  val classify2: [][]t -> [][]t

}


------ SOFTMAX -------- stable version
module softmax_stable (R:real): classification with t = R.t = {

  type t = R.t

  let classify  (x: []t) =
      let maxval = R.maximum x
      let exps = map (\x -> R.(exp (x - maxval))) x
      let sumexps = R.(reduce (+) R.(i32 0) exps)
      in  map (\x -> R.(x / sumexps)) exps

  let classify2 (X:[][]t) =
      map (\x -> classify x) X
}

-- module type classification_funcs = {

--   type t
--   val calc_classification: []t -> i32 -> []t

-- }


-- module class_funcs_coll (R:real) : classification_funcs with t = R.t = {

--   type t = R.t
--   module softmax = softmax_stable R

--   let calc_classification [m] (data: [m]t) (class_id: i32) =
--     if class_id == 1 then softmax.classify data
--     else data

-- }


module type loss = {

  type t
  val derivative: [][]t -> [][]t -> [][]t
}

module softmax_cross_entropy_with_logits (R:real) : loss with t = R.t = {

  type t = R.t
  module softmax = softmax_stable R

  let derivative [d] (labels: [d][]R.t) (logits:[d][]R.t) =
     map2 (\xr yr -> map2 (\x y  -> R.(y - x)) xr yr) labels (softmax.classify2 logits)
}
