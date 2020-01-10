import "../nn_types"

module type optimizer_type = {

  type t
  type ^learning_rate

  -- | Train function with signature
  --   network -> learning_rate -> input data -> labels
  --   -> batch_size -> classifier
  --   Returns the new network with updated weights
  val train [K] 'i 'w 'g 'e2 'o :
    NN i w o g o e2 (apply_grad3 t) ->
    learning_rate ->
    (input: [K]i) ->
    (labels: [K]o) ->
    (seed:i32) ->
    loss_func o t ->
    NN i w o g o e2 (apply_grad3 t)
}
