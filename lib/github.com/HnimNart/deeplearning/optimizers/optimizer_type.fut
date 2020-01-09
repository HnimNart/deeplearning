import "../nn_types"

module type optimizer_type = {

  type t
  type ^learning_rate

  -- | Train function with signature
  --   network -> learning_rate -> input data -> labels
  --   -> batch_size -> classifier
  --   Returns the new network with updated weights
  val train [n][m][K] 'i 'w 'g 'e2 'o :
    NN ([n]i) w ([m]o) g ([m]o) e2 (apply_grad3 t) ->
    learning_rate ->
    (input: [K][n]i) ->
    (labels: [K][m]o) ->
    (seed:i32) ->
    loss_func ([m]o) t ->
    NN ([n]i) w ([m]o) g ([m]o) e2 (apply_grad3 t)
}
