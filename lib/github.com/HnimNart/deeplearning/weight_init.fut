-- | Module for creating random numbers
--   Used for weight initilization
import "../../diku-dk/cpprandom/random"

module weight_initializer (R:real) : {

  type t = R.t
  ---- Using xavier uniform initializer [-limit, limit]
  ---- with limit = sqrt(6/(fan_in + fan_out))
  val gen_random_array_2d_xavier_uni: (i32, i32) -> i32 -> [][]t

  --- Using xavier norm initialize
  --- with mean = 0 and std = sqrt(2 / (fan_in + fan_out))
  val gen_random_array_2d_xavier_norm: (i32, i32) -> i32 -> [][]t

} = {

  type t = R.t

  module norm = normal_distribution R minstd_rand
  module uni  = uniform_real_distribution R minstd_rand

  let gen_rand_uni (i:i32)  (dist:(uni.num.t, uni.num.t)): t =
    let rng = uni.engine.rng_from_seed [i]
    let (_, x) = uni.rand dist rng in x

  let gen_random_array_uni (d: i32) (dist) (seed: i32) : []t =
    map (\x -> gen_rand_uni x dist) (map (\x -> x + d + seed) (iota d))

  let gen_random_array_2d_xavier_uni ((m,n):(i32, i32)) (seed:i32) : [][]t =
    let d = R.(((sqrt((i32 6)) / sqrt(i32 n + i32 m))) )
    let arr = gen_random_array_uni (m*n) (R.(negate d),d) seed
    in unflatten n m arr

  let gen_rand_norm (i: i32) (dist) : t =
    let rng = norm.engine.rng_from_seed [i]
    let (_, x) = norm.rand dist rng in x

  let gen_random_array_norm (d: i32) (seed: i32) (dist) : []t =
    map (\x -> gen_rand_norm x dist) (map (\x -> x + d + seed) (iota d))

  let gen_random_array_2d_xavier_norm ((m,n):(i32, i32)) (seed:i32) : [][]t =
    let n_sqrt = R.(sqrt (i32 2/ (i32 m + i32 n)))
    let dist = {mean = R.(i32 0), stddev = n_sqrt }
    let arr = gen_random_array_norm (m*n) seed dist
    in unflatten n m (map (\x -> R.(x)) arr)
}
