import "../nn_types"

module type trainer = {

  type t
  type ^alpha
  type loss_func 'o = {f:o -> o -> t, fd:o -> o -> o}

  -- | Train function with signature
  --   network -> learning_rate -> input data -> labels -> batch_size -> classifier
  --   Returns the new network with updated weights
  val train 'i 'w 'g 'e2 'o : NN ([]i) w ([]o) g ([]o) e2 (apply_grad t) ->
                              alpha ->
                              ([]i) ->
                              ([]o) ->
                              i32 ->
                              loss_func o ->
                              NN ([]i) w ([]o) g ([]o) e2 (apply_grad t)

}
