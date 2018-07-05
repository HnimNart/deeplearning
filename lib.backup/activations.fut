import "util"


module type activation_function = {

  type t
  val func: []t -> []t
  val derivative : []t -> []t

}

-- relu6
-- Nice to have activation functions
-- softplus
-- dropout
-- softsign
-- selu
-- elu

module Identity (R:real) : activation_function with t = R.t = {

  type t = R.t

  let func [m] (X:[m]t) =
    X

  let derivative [m] (X:[m]t) =
    map (\_ -> R.(i32 1)) X
}

-- SIGMOID ----------
module Sigmoid (R: real) : activation_function  with t = R.t = {

   type t = R.t

   let multMatrix [m] (X: [m]t) (Y:[m]t) : [m]t =
     map2 (\x y -> R.(x * y))  X Y

   let func [m] (X: [m]t) =
     map (\xc -> R.(i32 1 /(i32 1 + exp (negate xc)))) X

   let derivative [m] (X: [m]t) =
    map (\xc -> R.( xc * ((i32 1) - xc))) X

   let derivative' [m] (X: [m]t) =
    let X' = func X
    in map (\xc -> R.( xc * ((i32 1) - xc))) X'


}

-------- RELU ----------
module Relu (R: real): activation_function with t = R.t= {

  type t = R.t

  let func [m](X : [m]t) =
    map (\x -> R.(if x < i32 0 then i32 0 else x)) X

  let derivative [m] (X: [m]t) =
    map (\x -> R.(if x <=  i32 0 then i32 0 else i32 1)) X
}

----- TANH ---------
module Tanh (R:real): activation_function with t = R.t = {

   type t = R.t
   let tanhsingle (x: t) =
     R.((exp x - exp(negate x)) / ((exp x) + exp (negate x)))

   let func [m] (X: [m]t) =
     map (\x ->  R.(tanhsingle x)) X

   let derivative [m] (X: [m]t) =
     map (\x -> R.(i32 1 - tanhsingle(x)**(i32 2))) X
}

module type activation_funcs = {

  type t
  val calc_activation: []t -> i32 -> []t
  val calc_derivative: []t -> i32 -> []t

}

module activation_funcs_coll (R:real) : activation_funcs with t = R.t = {

  type t = R.t
  module identity = Identity R
  module tanh = Tanh R
  module sigmoid = Sigmoid R
  module relu = Relu R

  let calc_activation [m] (data: [m]t) (func: i32) =
    if func == 0 then data
    else if func == 1 then relu.func data
    else if func == 2 then sigmoid.func data
    else if func == 3 then tanh.func data
    else data

  let calc_derivative [m] (data:[m]t) (func: i32) =
    if func == 0 then identity.derivative data
    else if func == 1 then relu.derivative data
    else if func == 2 then sigmoid.derivative data
    else if func == 3 then tanh.derivative data
    else identity.derivative data
}
