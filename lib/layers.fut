import "types"
import "activations"






module type layer = {

  type input
  type weights
  type output
  type error_input
  type error_output

  val layer:

}


module dense (R:real) : layer with t = R.t = {








}



module layers (R:real) :{


  type act_pair_1d
  type dense
  val dense:  (i32, i32) -> act_pair_1d -> dense





} = {








}