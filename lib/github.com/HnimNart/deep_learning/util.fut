-- | Common utility functions used throughout
--   the deep learning library

module utility (R:real) : {

  type t = R.t

  --- Subtracts element wise
  val sub_v  : []t -> []t -> []t
  val sub_matrix:  [][]t -> [][]t -> [][]t

  --- Scales matrix/vector with t
  val scale_v: []t -> t -> []t
  val scale_matrix: [][]t -> t -> [][]t

  --- element wise product
  val hadamard_prod_2d : [][]t -> [][]t -> [][]t
  val hadamard_prod_3d : [][][]t -> [][][]t -> [][][]t
  val hadamard_prod_4d : [][][][]t -> [][][][]t -> [][][][]t

  --- Creates an diagonal matrix from vector
  --- with zeros for (i != j)
  val diag: []t -> [][]t
  --- Extracts diagonal from matrix
  val extract_diag: [][]t -> []t

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
    let retval = unsafe map (\i -> X_flat[i] ) (index)
    in retval

}
