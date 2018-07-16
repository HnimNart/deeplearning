import "../nn_types"

module type trainer = {

  type t

  -- | Train function with signature
  --   network -> learning_rate -> input data -> labels -> batch_size -> classifier
  --   Returns the new network with updated weights
  val train 'i 'w 'g 'e2 'o : NN ([]i) w ([]o) g ([]o) e2 (apply_grad t) -> t -> ([]i) -> ([]o) -> i32 -> (o -> o -> t, o -> o -> o) ->  NN ([]i) w ([]o) g ([]o) e2 (apply_grad t)

}
