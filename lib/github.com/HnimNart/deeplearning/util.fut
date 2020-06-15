-- | Common utility functions used throughout
--   the deep learning library

module utility (R:real) : {

  type t = R.t

  --- Subtracts element wise
  val sub_v [n] : [n]t -> [n]t -> [n]t
  val sub_matrix [n][m]:  [n][m]t -> [n][m]t -> [n][m]t

  --- Scales matrix/vector with t
  val scale_v [n]: [n]t -> t -> [n]t
  val scale_matrix [n][m]: [n][m]t -> t -> [n][m]t

  --- element wise product
  val hadamard_prod_2d [m][n] : [m][n]t -> [m][n]t -> [m][n]t
  val hadamard_prod_3d [m][n][p] : [m][n][p]t -> [m][n][p]t -> [m][n][p]t
  val hadamard_prod_4d [m][n][p][q] : [m][n][p][q]t -> [m][n][p][q]t -> [m][n][p][q]t

  --- Creates an diagonal matrix from vector
  --- with zeros for (i != j)
  val diag [n] : [n]t -> [n][n]t
  --- Extracts diagonal from matrix
  val extract_diag [n]: [n][n]t -> [n]t

} = {

  type t  = R.t

  let sub_v [d] (x: [d]t) (y: [d]t) : [d]t =
    map2 (\x y -> R.(x - y)) x y

  let sub_matrix [m][n] (X: [m][n]t) (Y:[m][n]t) : [m][n]t =
    map2 (\xr yr -> map2 (\x y -> R.(x - y)) xr yr) X Y

  let scale_v [d] (x: [d]t) (a: t): [d]t =
    map (\x -> R.(x * a) ) x

  let scale_matrix [m][n] (X: [m][n]t) (s:t) : [m][n]t =
    map (\x -> scale_v x s) X

  let hadamard_prod_2d [m][n] (X: [m][n]t) (Y:[m][n]t) : [m][n]t =
    map2 (\xr yr -> map2 (\x y -> R.(x * y)) xr yr) X Y

  let hadamard_prod_3d  [m][n][d] (X: [m][n][d]t)
                                  (Y:[m][n][d]t) : [m][n][d]t =
    map2 (\x y -> hadamard_prod_2d x y) X Y

  let hadamard_prod_4d  [m][n][p][q] (X: [m][n][p][q]t)
                                     (Y:[m][n][p][q]t) : [m][n][p][q]t =
    map2 (\x y -> hadamard_prod_3d x y) X Y


  let diag [len] (X:[len]t) =
    let elem = len ** 2
    let index  = map (\x -> x * len + x) (0..<len)
    let retval = scatter (replicate elem R.(i32 0)) index X
   in unflatten len len retval

  let extract_diag [len] (X:[len][len]t) =
    let X_flat = flatten X
    let index = map (\x -> x * len + x) (0..<len)
    let retval = map (\i -> X_flat[i] ) (index)
    in retval

}
