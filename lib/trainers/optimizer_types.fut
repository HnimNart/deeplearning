import "../nn_types"

module type trainer = {

  type t

  type ^updater

  val train 'i 'w 'g 'e2 'o : NN ([]i) w ([]o) g ([]o) e2 updater -> t -> ([]i) -> ([]o) -> i32 -> (o -> o -> t, o -> o -> o)
                             ->  NN ([]i) w ([]o) g ([]o) e2 updater

}
