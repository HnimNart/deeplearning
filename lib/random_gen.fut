-- | Module for creating random numbers in different ways
--   Used for weight initilization
import "/futlib/random"

module type random_generator = {

  type t
  val gen_random_array: i32 -> i32 -> []t
  val gen_random_array_2d: (i32, i32) -> i32 -> [][]t
  val gen_random_array_2d_w_scaling: (i32, i32) -> i32 -> [][]t
  val gen_random_array_2d_w_scaling_conv: (i32, i32) -> i32 -> [][]t
  val gen_random_array_3d: (i32,i32, i32) -> i32 -> [][][]t

}


--- Generate random numbers from a standard normal distribution
--- Some provides scaling s.t. Var(X) = 1/sqrt(N_in)
module normal_random_array (R:real) : random_generator
                                      with t = R.t = {

  type t = R.t

  module dist = normal_distribution R minstd_rand
  let stdnorm = {mean = R.(i32 0), stddev = R.(i32 1)}

  let gen_rand (i: i32) : t =
    let rng = dist.engine.rng_from_seed [i]
    let (_, x) = dist.rand stdnorm rng in
    let retval =  R.(if isinf x then i32 0 else x) in retval

  let gen_random_array (d: i32) (seed: i32) : []t =
    map gen_rand (map (\x -> x + d + seed) (iota d))

  let gen_random_array_2d_w_scaling_conv ((m,n):(i32, i32)) (seed:i32) : [][]t =
    let n_sqrt = R.(sqrt (i32 m))
    let arr = gen_random_array (m*n) seed
    in unflatten n m (map (\x -> R.(x * n_sqrt)) arr)

  let gen_random_array_2d_w_scaling ((m,n):(i32, i32)) (seed:i32) : [][]t =
    let n_sqrt = R.(sqrt (i32 n))
    let arr = gen_random_array (m*n) seed
    in unflatten n m (map (\x -> R.(x / n_sqrt)) arr)

  let gen_random_array_2d ((m,n):(i32, i32)) (seed:i32) : [][]t =
    let arr = gen_random_array (m*n) seed
    in unflatten n m arr

  let gen_random_array_3d ((m,n,p):(i32, i32, i32)) (seed:i32) : [][][]t =
    map (\i -> gen_random_array_2d (m,n) (seed+i)) (0..<p)

}
