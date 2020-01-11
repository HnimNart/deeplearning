import "../nn_types"
import "layer_type"

type flatten_layer [m][a][b] [n] 't =
    NN ([m][a][b]t) () ([n]t)
       () ([n]t) ([m][a][b]t)
       (apply_grad3 t)

module flatten (R:real) : {
  type t = R.t
  val init : (m: i32) -> (a: i32) -> (b: i32) -> (n: i32)
          -> flatten_layer [m][a][b] [n] t
} = {
  type t = R.t

  let forward [k] [m][a][b] 't
              (n: i32) (training: bool) () (input: [k][m][a][b]t) : ([k](), [k][n]t) =
    (replicate k (), map (\image -> flatten_3d image : [n]t) input)

  let backward [k]
               (m: i32) (a: i32) (b: i32) (n: i32)
               (first_layer:bool)
               (_: apply_grad3 t)
               ()
               _
               (error: [k][n]t) : ([k][m][a][b]t, ()) =
    let error' = map (unflatten_3d m a b) error
    in (error', ())

  let init m a b n : flatten_layer [m][a][b] [n] t =
    {forward  = \_k -> forward n,
     backward = \_k -> backward m a b n,
     weights  = ()}
}
