




import "../lib/util"



module random = normal_random_array f32


let main =
  let n = 10000
  let a = random.gen_random_array n 1
  let mean_a = (reduce (+) 0.0 a) f32./ f32.(i32  n)
  let tmp    = map (\x -> (x-mean_a)**2) a
  let sum_tmp = reduce (+) 0.0 tmp
  in sum_tmp / f32.(i32 n - 1)