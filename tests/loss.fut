import "../lib/loss_funcs"

module loss = loss_funcs f64
-- ==
-- entry: cross_entropy
-- input{[ 0.228 ,0.619, 0.153 ]
--       [ 0.0, 1.0, 0.0 ] }
-- output{ 0.479 }

entry cross_entropy input labels = loss.cross_entropy.1 input labels

-- ==
-- entry: cross_entropy_deriv
-- input{[ 0.10650698 , 0.10650698, 0.78698604 ]
--       [ 0.0, 1.0, 0.0 ] }
-- output{ [-0.0, -9.3890561, -0.0 ] }
entry cross_entropy_deriv input labels = loss.cross_entropy.2 input labels

-- ==
-- entry: softmax_cross_entropy_with_logits
-- input {[1.0,2.0,3.0]
--        [0.0,0.0,1.0]}
-- output { 0.407606f64}

entry softmax_cross_entropy_with_logits input labels =
      loss.softmax_cross_entropy_with_logits.1 input labels

-- ==
-- entry: softmax_cross_entropy_with_logits_deriv
-- input {[1.0,2.0,3.0]
--        [0.0,0.0,1.0]}
-- output {[0.090031, 0.244728, -0.334759 ]}

entry softmax_cross_entropy_with_logits_deriv input labels =
      loss.softmax_cross_entropy_with_logits.2 input labels

-- ==
-- entry: sum_of_squares_error
-- input {[1.0,2.0,3.0]
--        [0.0,0.0,1.0]}
-- output { 4.5 }

entry sum_of_squares_error input labels =
     loss.sum_of_squares_error.1 input labels

-- ==
-- entry: sum_of_squares_error_deriv
-- input {[1.0,2.0,3.0]
--        [0.0,0.0,1.0]}
-- output { [ 1.0, 2.0, 2.0 ] }

entry sum_of_squares_error_deriv input labels =
     loss.sum_of_squares_error.2 input labels