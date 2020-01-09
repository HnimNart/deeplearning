import "../nn_types"

module type optimizer_type = {

  type t
  type ^learning_rate

  -- | Train function with signature
  --   network -> learning_rate -> input data -> labels
  --   -> batch_size -> classifier
  --   Returns the new network with updated weights
  val train 'i 'w 'g 'e2 'o [n] [m] [p] :
    NN ([n]i) w ([m]o) g ([p]o) e2 (apply_grad2 ([n]i) ([p]o)) ->
    learning_rate ->
    (input:([n]i)) ->
    (labels:([m]o)) ->
    (seed:i32) ->
    loss_func o t ->
    NN ([n]i) w ([m]o) g ([p]o) e2 (apply_grad2 ([n]i) ([p]o))
}
