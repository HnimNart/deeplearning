
module type classification = {

  type t
  val classify: []t -> []t

}


------ SOFTMAX -------- stable version
module softmax_stable (R:real): classification with t = R.t = {

  type t = R.t


  let classify  (x: []t) =
      let maxval = R.maximum x
      let exps = map (\x -> R.(exp (x - maxval))) x
      let sumexps = R.(reduce (+) R.(i32 0) exps)
      in  map (\x -> R.(x / sumexps)) exps

}

module type classification_funcs = {

  type t
  val calc_classification: []t -> i32 -> []t

}


module class_funcs_coll (R:real) : classification_funcs with t = R.t = {

  type t = R.t
  module softmax = softmax_stable R

  let calc_classification [m] (data: [m]t) (class_id: i32) =
    if class_id == 1 then softmax.classify data
    else data

}
