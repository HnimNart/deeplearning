import "../lib/github.com/HnimNart/deeplearning/weight_init"

module rand = weight_initializer f64

-- ==
-- entry: xavier_1
-- input  { [10, 10,1] }  output { true }
-- compiled input  { [1000, 1000,1] }  output { true }
-- compiled input  { [128, 256,1] }  output { true }
-- compiled input  { [10, 10,2] }  output { true }
-- compiled input  { [1000, 1000,2] }  output { true }
-- compiled input  { [128, 256,2] }  output { true }
-- compiled input  { [128, 256,3] }  output { true }

entry xavier_1 (input: []i32) =
  let (m,n,seed) = (input[0], input[1], input[2])
  let arr = flatten  (rand.gen_random_array_2d_xavier_uni (i64.i32 m) (i64.i32 n) seed)
  let lim = f64.((sqrt((i32 6)) / sqrt(i32 n + i32 m)))
  in ((f64.(minimum arr >= (negate lim)) && f64.(maximum arr <= lim) ))
