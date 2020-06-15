import "layer_type"
import "../nn_types"
import "../util"
import "../weight_init"
import "../../../diku-dk/linalg/linalg"

type^ dense_layer [m] [n] 't =
  NN ([m]t) (std_weights [n][m] [n] t) ([n]t)
     ([m]t, [n]t) ([n]t) ([m]t)
     (apply_grad3 t)

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
  let forward (k: i32)
              (n: i32) (m: i32)
              (act: [n]t -> [n]t)
              (_training:bool)
              ((w,b): std_weights [n][m] [n] t)
              (input: [k][m]t)
            : ([k]([m]t, [n]t), [k][n]t) =
    let res      = lalg.matmul w (transpose input)
    let res_bias = transpose (map2 (\xr b' -> map (\x -> (R.(x + b'))) xr) res b)
    let res_act  = map (\x -> act x) (res_bias)
    let cache    = zip input res_bias
    in (cache, res_act)

  -- Backward propagation
  let backward (k: i32)
               (n: i32) (m: i32)
               (act: [n]t -> [n]t)
               (_first_layer:bool)
               (apply_grads: apply_grad3 t)
               ((w,b): std_weights [n][m] [n] t)
               (cache: [k]([m]t, [n]t))
               (error: [k][n]t)
             : ([k][m]t, std_weights [n][m] [n] t) =
    let (input, inp_w_bias) = unzip cache
    let deriv    = (map (\x -> act x) inp_w_bias)
    let delta    = transpose (util.hadamard_prod_2d error deriv)
    let w_grad   = lalg.matmul delta input
    let b_grad   = map (R.sum) delta
    let (w', b') = apply_grads n m (w,b) (w_grad, b_grad)

    --- Calc error to backprop to previous layer
    let error' = transpose (lalg.matmul (transpose w) delta)
    in (error', (w', b'))


  let init m n (act: activation_func ([n]t)) (seed:i32) : dense_layer [m] [n] t =
    let w = w_init.gen_random_array_2d_xavier_uni m n seed
    let b = map (\_ -> R.(i32 0)) (0..<n)
    in
    {forward  = \k -> forward k n m act.f,
     backward = \k -> backward k n m act.fd,
     weights  = (w,b)}

}
