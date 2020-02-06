import "../lib/github.com/HnimNart/deeplearning/activation_funcs"


module act = activation_funcs f64

-- ==
-- entry: relu
-- input { [9.74f64, -8.64f64,
--         -8.88f64,  2.42f64,
--          5.34f64,  8.86f64,
--          5.63f64,  0.10f64,
--         -2.17f64,  7.69364] }
--
-- output{ [9.74f64, 0.00f64,
--          0.00f64, 2.42f64,
--          5.34f64, 8.86f64,
--          5.63f64, 0.107f64,
--          0.00f64, 7.69f64] }

entry relu [d] (input:[d]f64) = (act.Relu_1d d).f input

-- ==
-- entry: relu_derivative
-- input { [9.74f64, -8.64f64,
--         -8.88f64,  2.42f64,
--          5.34f64,  8.86f64,
--          5.63f64,  0.10f64,
--         -2.17f64,  7.69364] }
--
-- output { [1.00f64, 0.00f64,
--           0.00f64, 1.00f64,
--           1.00f64, 1.00f64,
--           1.00f64, 1.00f64,
--           0.00f64, 1.00f64] }

entry relu_derivative [d] (input:[d]f64) = (act.Relu_1d d).fd input

-- ==
-- entry: sigmoid
-- input  { [0.458f64, 1.0000f64, 0.0000f64] }
-- output { [0.613f64, 0.7311f64, 0.5000f64] }
entry sigmoid [d] (input:[d]f64) = (act.Sigmoid_1d d).f input

-- ==
-- entry: sigmoid_derivative
-- input  { [0.50000f64, 0.00000f64, 0.75000f64] }
-- output { [0.23500f64, 0.25000f64, 0.21789f64] }
entry sigmoid_derivative [d] (input:[d]f64)  = (act.Sigmoid_1d d).fd input

-- ==
-- entry: tanh
-- input  {[ 0.10000f64, 1.000000f64, -0.100000f64 ]}
-- output {[ 0.09967f64, 0.761594f64, -0.099668f64 ]}
entry tanh [d] (input:[d]f64) = (act.Tanh_1d d).f input

-- ==
-- entry: tanh_derivative
-- input  {[ 0.10000f64, 1.00000f64, -0.50000f64, 0.000000f64]}
-- output {[ 0.99006f64, 0.41997f64, 0.786448f64, 1.000000f64]}
entry tanh_derivative [d] (input:[d]f64) = (act.Tanh_1d d).fd input

-- ==
-- entry: identity
-- input  {[1.00f64, 3.00f64, -1.00f64]}
-- output {[1.00f64, 3.00f64, -1.00f64]}
entry identity [d] (input:[d]f64) = (act.Identity_1d d).f input

-- ==
-- entry: identity_derivative
-- input  {[1.00f64, 3.00f64, -1.00f64]}
-- output {[1.00f64, 1.00f64,  1.00f64]}
entry identity_derivative [d] (input:[d]f64) = (act.Identity_1d d).fd input

-- ==
-- entry: softmax
-- input  {[3.0000f64, 4.0000f64, 1.00000f64]}
-- output {[0.2594f64, 0.7053f64, 0.03511f64]}
entry softmax [d] (input:[d]f64) = (act.Softmax_1d d).f input

-- BROKEN! isn't correct
-- entry: softmax_derivative
-- input {[3f64,4f64, 1f64]}
-- output {[0.192158047412, 0.207817201944f64, 0.033885680905f64]}
-- entry softmax_derivative [d] (input:[d]f64) = act.Softmax_1d.fd input
