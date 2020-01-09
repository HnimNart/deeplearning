import "layer_type"
import "../nn_types"
import "../util"
import "../weight_init"
import "../../../diku-dk/linalg/linalg"

type dense_layer [m] [n] 't =
  NN ([n][m]t) (std_weights [n][m] [n] t) ([n][n]t)
     ([n][m]t, [n][n]t) ([n][n]t) ([n][m]t)
     (apply_grad2 ([n][m]t) ([n]t))

-- | Fully connected layer
module dense (R:real) : { type t = R.t
                          val init : (m: i32) -> (n: i32)
                                  -> activation_func ([n]t)
                                  -> i32
                                  -> dense_layer [m] [n] t
                         } = {

  type t            = R.t

  module lalg   = mk_linalg R
  module util   = utility R
  module w_init = weight_initializer R

  -- Forward propagation
  let forward [n][m]
              (act:[n]t -> [n]t)
              (training:bool)
              ((w,b): std_weights [n][m] [n] t)
              (input: [n][m]t)
            : (([n][m]t, [n][n]t), [n][n]t) =
    let res      = lalg.matmul w (transpose input)
    let res_bias = transpose (map2 (\xr b' -> map (\x -> (R.(x + b'))) xr) res b)
    let res_act  = map (\x -> act x) (res_bias)
    let cache    = (input, res_bias)
    in (cache, res_act)

  -- Backward propagation
  let backward [n][m]
               (act: [n]t -> [n]t)
               (first_layer:bool)
               (apply_grads: apply_grad [n][m] [n] t)
               ((w,b): std_weights [n][m] [n] t)
               ((input, inp_w_bias): ([n][m]t, [n][n]t))
               (error: [n][n]t)
             : ([n][m]t, std_weights [n][m] [n] t) =

    let deriv    = (map (\x -> act x) inp_w_bias)
    let delta    = transpose (util.hadamard_prod_2d error deriv)
    let w_grad   = lalg.matmul delta input
    let b_grad   = map (R.sum) delta
    let (w', b') = apply_grads (w,b) (w_grad, b_grad)

    --- Calc error to backprop to previous layer
    let error' = transpose (lalg.matmul (transpose w) delta)
    in (error', (w', b'))


  let init m n (act: activation_func ([n]t)) (seed:i32) : dense_layer [m] [n] t =
    let w = w_init.gen_random_array_2d_xavier_uni m n seed
    let b = map (\_ -> R.(i32 0)) (0..<n)
    in
    {forward  = forward act.f,
     backward = backward act.fd,
     weights  = (w,b)}

}
