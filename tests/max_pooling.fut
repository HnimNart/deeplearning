
import "../lib/layers/max_pooling"


module max = max_pooling_2d f32



let max_layer = max.init (2,2) ((),()) 0


let a:[][][][]f32 = [[[[23,4,16,90],[12,32,12,45],[5,7,8,9],[2,12,14,56]]]]


let main =
  let (f,b,_,w) = max_layer
  let (os, output) = f true w a
  let (err, _) = b w os output in err
