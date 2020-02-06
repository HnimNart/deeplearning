import "../lib/github.com/HnimNart/deeplearning/loss_funcs"

module loss = loss_funcs f64
-- ==
-- entry: cross_entropy
-- input{[ 0.228f64 ,0.619f64, 0.153f64 ]
--       [ 0.000f64, 1.000f64, 0.000f64 ] }
-- output{ 0.479f64 }

entry cross_entropy [n] input (labels: [n]f64) =
  (loss.cross_entropy n).f input labels

-- ==
-- entry: cross_entropy_deriv
-- input  { [ 0.10650698f64,  0.10650698f64,  0.78698604f64 ]
--          [ 0.00000000f64,  1.00000000f64,  0.00000000f64 ] }
-- output { [-0.00000000f64, -9.38905610f64, -0.00000000f64 ] }
entry cross_entropy_deriv [n] input (labels: [n]f64) =
  (loss.cross_entropy n).fd input labels

-- ==
-- entry: softmax_cross_entropy_with_logits
-- input {[1.000f64, 2.000f64, 3.000f64]
--        [0.000f64, 0.000f64, 1.000f64]}
-- output { 0.407606f64}

entry softmax_cross_entropy_with_logits [n] input (labels: [n]f64) =
  (loss.softmax_cross_entropy_with_logits n).f input labels

-- ==
-- entry: softmax_cross_entropy_with_logits_deriv
-- input { [1.0000f64, 2.0000f64, 3.00000f64]
--         [0.0000f64, 0.0000f64, 1.00000f64]}
-- output {[0.090031f64, 0.244728f64, -0.334759f64 ]}

entry softmax_cross_entropy_with_logits_deriv [n] input (labels: [n]f64) =
  (loss.softmax_cross_entropy_with_logits n).fd input labels

-- ==
-- entry: sum_of_squares_error
-- input {[1.000f64, 2.0000f64, 3.0000f64]
--        [0.000f64, 0.0000f64, 1.0000f64]}
-- output { 4.5000f64 }

entry sum_of_squares_error [n] input (labels: [n]f64) =
  (loss.sum_of_squares_error n).f input labels

-- ==
-- entry: sum_of_squares_error_deriv
-- input {  [1.000f64, 2.0000f64, 3.0000f64]
--          [0.000f64, 0.0000f64, 1.0000f64] }
-- output { [1.0000f64, 2.0000064, 2.000064 ] }

entry sum_of_squares_error_deriv [n] input (labels: [n]f64) =
  (loss.sum_of_squares_error n).fd input labels
