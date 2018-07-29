import "../lib/weight_init"

module rand = weight_initializer f64

-- ==
-- entry: xavier_1
-- input  { [10, 10] }  output { true }
-- input  { [1000, 1000] }  output { true }
-- input  { [128, 256] }  output { true }

entry xavier_1 input =
  let (m,n) = (input[0], input[1])
  let arr = flatten  (rand.gen_random_array_2d_xavier_uni (m,n) 1)
  let lim = f64.((sqrt((i32 6)) / sqrt(i32 n + i32 m)))
  in ((f64.(minimum arr >= (negate lim)) && f64.(minimum arr <= lim) ))
