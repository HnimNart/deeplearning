import "/futlib/random"
import "/futlib/linalg"


---------------- Random num gen ----------------------
module type random_generator = {

 type t
  val gen_random_array: i32 -> i32 -> []t
  val gen_random_array_2d: (i32, i32) -> i32 -> [][]t
  val gen_random_array_2d_w_scaling: (i32, i32) -> i32 -> [][]t

  val gen_random_array_2d_w_scaling_conv: (i32, i32) -> i32 -> [][]t
  val gen_random_array_3d: (i32,i32, i32) -> i32 -> [][][]t

}

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

module utility (R:real) : {

  type t = R.t

  val mult_matrix: [][]t -> [][]t -> [][]t
  val sub_matrix:  [][]t -> [][]t -> [][]t
  val add_matrix:  [][]t -> [][]t -> [][]t
  val scale_matrix: [][]t -> t -> [][]t

  val mult_v : []t -> []t -> []t
  val sub_v  : []t -> []t -> []t
  val scale_v: []t -> t -> []t

  val diag: []t -> [][]t
  val extract_diag: [][]t -> []t

} = {

  type t  = R.t

  let mult_v [m] (X:[m]t) (Y:[m]t) : [m]t =
    map2 (\x y -> R.(x * y)) X Y

  let scale_v [d] (x: [d]t) (a: t): [d]t =
    map (\x -> R.(x * a) ) x

  let sub_v [d] (x: [d]t) (y: [d]t) : [d]t =
    map2 (\x y -> R.(x - y)) x y

  -- Element wise
  let mult_matrix  [m][n] (X: [m][n]t) (Y:[m][n]t) : [m][n]t =
    map2 (\xr yr -> map2 (\xc yc -> R.(xc * yc)) xr yr) X Y

  let sub_matrix [m][n] (X: [m][n]t) (Y:[m][n]t) : [m][n]t =
    map2 (\xr yr -> map2 (\x y -> R.(x - y)) xr yr) X Y

  let add_matrix (X:[][]t) (Y:[][]t) =
    map2 (\xr yr -> map2 (\x y -> R.(x + y)) xr yr) X Y

  let scale_matrix [m][n] (X: [m][n]t) (s:t) : [m][n]t =
    map (\x -> scale_v x s) X

  let diag (X:[]t) =
    let len = length X
    let elem = len ** 2
    let index  = map (\x -> x * len + x) (0..<len)
    let retval = scatter (replicate elem R.(i32 0)) index X
   in unflatten len len retval


  let extract_diag (X:[][]t) =
    let len = length X
    let X_flat = flatten X
    let index = map (\x -> x * len + x) (0..<len)
    let retval = map (\i -> X_flat[i] ) (index)
     in retval

}
