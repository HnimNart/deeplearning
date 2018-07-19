import "../lib/tensorflow"
import "/futlib/linalg"
module tf = tensorflow f32
module lalg = linalg f32

let conv = tf.layers.Conv2d (2, 2, 1, 2 ) tf.nn.identity 1

let input1:[][][]f32 = [[[1,2,3],[4,5,6],[7,8,9]], [[10,11,12],[13,14,15],[16,17,18]]]
let input2:[][][]f32 = [[[10,11,12],[13,14,15],[16,17,18]], [[10,11,12],[13,14,15],[16,17,18]]]

let w:[][]f32 = [[1,2,3,4, 5,6,7,8],[8,7,6,5,4,3,2,1]]
