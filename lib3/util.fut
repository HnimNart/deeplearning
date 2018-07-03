import "/futlib/random"
import "/futlib/linalg"

-------- UTILITY ------------
let multMatrix [m][n] (X: [m][n]f64) (Y:[m][n]f64) : [m][n]f64 =
  map2 (\xr yr -> map2 (\xc yc -> xc * yc) xr yr) X Y

let subMatrix [m][n] (X: [m][n]f64) (Y:[m][n]f64) : [m][n]f64 =
  map2 (\xr yr -> map2 (\xc yc -> xc - yc) xr yr) X Y

let addMatrix [m][n] (X: [m][n]f64) (Y:[m][n]f64) : [m][n]f64 =
  map2 (\xr yr -> map2 (\xc yc -> xc + yc) xr yr) X Y


let subV [d] (x : [d]f64) (y: [d]f64): [d]f64  =
  map2 (-) x y

let multV [m] (X: [m]f64) (Y: [m]f64) =
  map2 (*) X Y

let logV [m] (X: [m]f64) =
  map (\x -> f64.log x) X

let negV [d] (x: [d]f64) : [d]f64 =
  map (0-) x

let addbias [m][n][q] (X: [m][n]f64) (Y: [q][n]f64) =
  map (\x -> (map2 (+) x Y[0])) X

let scaleV [d] (x: [d]f64) (a: f64): [d]f64 =
  map (*a) x

let scaleMatrix [m][n] (X: [m][n]f64) (s:f64) : [m][n]f64 =
  map (\x -> scaleV x s) X

let divideMatrix [m][n] (X: [m][n]f64) (s:f64) : [m][n]f64 =
  map (\xr -> map (\y -> y / s) xr) X

let sumfstCol [m][n] (X : [m][n]f64) =
  reduce (+) 0.0 X[0]

let getCol [m][n] (M: [m][n]f64) (i: i32) : [m]f64 =
  map (\x -> x[i] ) M

let abovezero [m][n] (X: [m][n]f64) =
  map (\xr -> map (\y -> if y < 0 then 0 else y) xr) X

---- Create diagonal matrix
let diag2d [n] (x: [n]f64) =
  let lenarr = length x
  let index = 0
  let lensquared = lenarr**2
  let retval = replicate lensquared 0.0
  in let (retval , _ ) = loop (retval, index) for i < lenarr**2  do
     if i % lenarr == 0 then
        let retval[i + index] = x[index] in
        (retval, index + 1)
     else
        (retval,  index)
  in unflatten lenarr lenarr retval


-- let find_idx_first [n] (e:i32) (xs:[n]i32) : i32 =
--   let es = map2 (\x i -> if x==e then i else n) xs (iota n)
--   let res = reduce min n es
--   in if res == n then -1 else res


---------------- Random num gen ----------------------
module type random_generator = {

 type t
 val gen_random_array: i32 -> []t
}

let seed = 1
module normal_random_array (R:real) : random_generator
                                      with t = R.t = {

  type t = R.t

  module dist = normal_distribution R minstd_rand
  let stdnorm = {mean = R.(i32 0), stddev = R.(i32 1)}
  let gen_rand (i: i32) : t =
    let rng = dist.engine.rng_from_seed [i]
    let (_, x) = dist.rand stdnorm rng in
    R.(x / i32 100)

  let gen_random_array (d: i32)  : []t =
    map gen_rand (map (\x -> x + d + seed) (iota d))

    -- let rng = dist.engine.rng_from_seed [seed]
    -- let tmp2 = replicate d (R.(i32 0))
    -- let tmp = replicate d rng
    -- let s = scan ( \(r,_) (_,_) -> dist.rand stdnorm r) (rng, R.(i32 0)) (zip tmp tmp2)
    -- let (_, res) = unzip s
    -- in map (\x -> R.(x / i32 100)) res
}

---- Usefull funcs ------
module type util = {

  type t
  val multMatrix: [][]t -> [][]t -> [][]t
  val multV: []t -> []t -> []t
  val addbias: []t -> []t -> []t
  val scaleV : []t -> t -> []t
  val subV : []t -> []t -> []t
  val scaleMatrix: [][]t -> t -> [][]t
  val subMatrix: [][]t -> [][]t -> [][]t
}

module utility_funcs (R:real) : util with t = R.t = {

  type t  = R.t
  -- Element wise
  let multMatrix  [m][n] (X: [m][n]t) (Y:[m][n]t) : [m][n]t =
    map2 (\xr yr -> map2 (\xc yc -> R.(xc * yc)) xr yr) X Y

  let multV [m] (X:[m]t) (Y:[m]t) : [m]t =
    map2 (\x y -> R.(x * y)) X Y

  let addbias [m] (X:[m]t) (b:[m]t) =
    map2 (\x y -> R.(x + y) ) X b

  let scaleV [d] (x: [d]t) (a: t): [d]t =
    map (\x -> R.(x * a) ) x

  let scaleMatrix [m][n] (X: [m][n]t) (s:t) : [m][n]t =
    map (\x -> scaleV x s) X

  let subV [d] (x: [d]t) (y: [d]t) : [d]t =
    map2 (\x y -> R.(x - y)) x y

  let subMatrix [m][n] (X: [m][n]t) (Y:[m][n]t) : [m][n]t =
    map2 (\xr yr -> map2 (\x y -> R.(x - y)) xr yr) X Y


}
