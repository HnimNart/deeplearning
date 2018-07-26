import "../lib/activation_funcs"

module act = activation_funcs f64

-- ==
-- entry: relu
-- input { [9.742936306782301f64, -8.647782872026834f64,
--          -8.881675545071436f64, 2.427798292646628f64,
--          5.340107228997773f64, 8.869733881800375f64,
--          5.639354659169356f64, 0.10718960311773174f64,
--          -2.178159317015176f64, 7.693139343115941f64] }
--
-- output{ [9.742936306782301f64, 0.000000000000000f64,
--          0.000000000000000f64, 2.427798292646628f64,
--          5.340107228997773f64, 8.869733881800375f64,
--          5.639354659169356f64, 0.10718960311773174f64,
--          0.00000000000000f64,  7.693139343115941f64]}

entry relu [d] (input:[d]f64) = act.Relu_1d.f input

-- ==
-- entry: relu_derivative
-- input  {[9.742936306782301f64, -8.647782872026834f64,
--          -8.881675545071436f64, 2.427798292646628f64,
--          5.340107228997773f64,  8.869733881800375f64,
--          5.639354659169356f64,  0.107189603117731f64,
--          -2.178159317015176f64, 7.693139343115941f64] }
--
-- output {[1.00, 0.00,
--          0.00, 1.00,
--          1.00, 1.00,
--          1.00, 1.00,
--          0.00, 1.00] }

entry relu_derivative [d] (input:[d]f64) = act.Relu_1d.fd input

-- ==
-- entry: sigmoid
-- input  { [0.458f64] }
-- output { [0.61253961344091512f64] }
entry sigmoid [d] (input:[d]f64) = act.Sigmoid_1d.f input

-- ==
-- entry: sigmoid_derivative
-- input  { [0.5f64, 0.0f64]}
-- output { [0.2350037122015945f64, 0.25f64] }
entry sigmoid_derivative [d] (input:[d]f64)  = act.Sigmoid_1d.fd input

-- ==
-- entry: tanh
-- input  {[ 0.1f64, 1.0f64, -0.1f64 ]}
-- output {[0.099668f64, 0.761594f64, -0.099668f64 ]}
entry tanh [d] (input:[d]f64) = act.Tanh_1d.f input

-- ==
-- entry: tanh_derivative
-- input {[ 0.1f64, 1.0f64, -0.5f64, 0.0f64 ]}
-- output {[0.990066f64, 0.419974f64, 0.786448f64, 1.000000f64]}
entry tanh_derivative [d] (input:[d]f64) = act.Tanh_1d.fd input

-- ==
-- entry: identity
-- input {[1f64, 3f64, -1f64]}
-- output {[1f64, 3f64, -1f64]}
entry identity [d] (input:[d]f64) = act.Identity_1d.f input

-- ==
-- entry: identity_derivative
-- input {[1f64, 3f64, -1f64]}
-- output {[1f64, 1f64, 1f64]}
entry identity_derivative [d] (input:[d]f64) = act.Identity_1d.fd input

-- ==
-- entry: softmax
-- input {[3f64,4f64, 1f64]}
-- output {[0.25949646034242f64, 0.70538451269824f64, 0.03511902695924f64]}
entry softmax [d] (input:[d]f64) = act.Softmax_1d.f input

-- BROKEN! isn't correct
-- entry: softmax_derivative
-- input {[3f64,4f64, 1f64]}
-- output {[0.192158047412, 0.207817201944f64, 0.033885680905f64]}
-- entry softmax_derivative [d] (input:[d]f64) = act.Softmax_1d.fd input