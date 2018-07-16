-- | Common utility functions used throughout
--   the deep learning code

module utility (R:real) : {

  type t = R.t

  --- all vector and matrix functions are
  --- element wise
  val mult_v : []t -> []t -> []t
  val sub_v  : []t -> []t -> []t
  val scale_v: []t -> t -> []t

  val mult_matrix: [][]t -> [][]t -> [][]t
  val sub_matrix:  [][]t -> [][]t -> [][]t
  val add_matrix:  [][]t -> [][]t -> [][]t
  val scale_matrix: [][]t -> t -> [][]t

  val mult_matrix_3d: [][][]t -> [][][]t -> [][][]t

  val mult_matrix_4d: [][][][]t -> [][][][]t -> [][][][]t

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

  let mult_matrix_3d  [m][n][d] (X: [m][n][d]t) (Y:[m][n][d]t) : [m][n][d]t =
    map2 (\X1 Y1 -> map2 (\xr yr -> map2 (\xc yc -> R.(xc * yc)) xr yr) X1 Y1) X Y

  let mult_matrix_4d  [m][n][p][q] (X: [m][n][p][q]t) (Y:[m][n][p][q]t) : [m][n][p][q]t =
    map2 (\x y -> mult_matrix_3d x y) X Y

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
    let retval = unsafe map (\i -> X_flat[i] ) (index)
    in retval

}
