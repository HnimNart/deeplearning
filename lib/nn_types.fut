-- | Network types



--- Network definition
type forwards   'input 'w 'output 'garbage = bool -> w -> input -> (garbage, output)
type backwards  'g 'w  'err_in  'err_out   = w -> g ->  err_in  -> (err_out, w)
type update     'w '^f                     = f -> w -> w -> w

type NN 'input 'w 'output 'g 'e_in 'e_out '^f = (forwards input w output g,
                                                 backwards g w e_in e_out,
                                                 update w f,
                                                 w)
--- The 'standard' weight definition
--- used by optimizers
type std_weights 't =  ([][]t, []t)
type apply_grad 't  = std_weights t -> std_weights t -> std_weights t

--- Commonly used types
type arr1d 't = []t
type arr2d 't = [][]t
type arr3d 't = [][][]t
type arr4d 't = [][][][]t

--- Function pairs
--- Used for activation functions
--- Denotes a function and it's derivative
type f_pair_1d 't = ([]t -> []t, []t-> []t)
type f_pair_2d 't = ([][]t -> [][]t, [][]t-> [][]t)


--- Layer types
--- Dense
type dense_tp 't =
    NN (arr2d t) (arr2d t,arr1d t) (arr2d t) ((arr2d t, arr2d t)) (arr2d t) (arr2d t) (apply_grad t)
--- Conv2d
type conv2d_tp 't =
    NN (arr4d t) (arr2d t,arr1d t) (arr4d t) ((i32, i32, i32), arr3d  t,arr4d  t) (arr4d  t) (arr4d  t) (apply_grad t)
--- Max pooling
type max_pooling_tp 't = NN (arr4d t) () (arr4d t) (arr4d (i32, i32)) (arr4d t) (arr4d t) (apply_grad t)
--- Flatten
type flatten_tp 't =  NN (arr4d t) () (arr2d t) (i32, i32, i32) (arr2d t) (arr4d t) (apply_grad t)
