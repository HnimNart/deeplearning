-- | Network types
type forwards   'input 'w 'output 'cache = bool -> w -> input -> (cache, output)
type backwards  'c 'w  'err_in  'err_out = bool -> w -> c ->  err_in  -> (err_out, w)
type update     'w '^f                   = f -> w -> w -> w

type NN 'input 'w 'output 'c 'e_in 'e_out '^f = (forwards input w output c,
                                                 backwards c w e_in e_out,
                                                 update w f,
                                                 w)

--- Commonly used types
type arr1d 't = []t
type arr2d 't = [][]t
type arr3d 't = [][][]t
type arr4d 't = [][][][]t

type dims2d  = (i32, i32)
type dims3d  = (i32, i32, i32)

--- The 'standard' weight definition
--- used by optimizers
type std_weights 't =  ([][]t, []t)
type apply_grad 't  = std_weights t -> std_weights t -> std_weights t

--- Function pairs
--- Denotes a function and it's derivative
type f_pair_1d 't = ([]t -> []t, []t-> []t)

type loss_pair_1d 't = ([]t -> []t -> t, []t -> []t -> []t )
