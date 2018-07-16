-- | Network types



--- Network definition
type forwards   'input 'w 'output 'garbage = bool -> w -> input -> (garbage, output)
type backwards  'g 'w  'err_in  'err_out   = w -> g ->  err_in  -> (err_out, w)
type update     'w '^f                     = f -> w -> w -> w

type NN 'input 'w 'output 'g 'e_in 'e_out '^f = (forwards input w output g,
                                                 backwards g w e_in e_out,
                                                 update w f,
                                                 w)
--- Commonly used types
type arr1d 't = []t
type arr2d 't = [][]t
type arr3d 't = [][][]t
type arr4d 't = [][][][]t

--- The 'standard' weight definition
--- used by optimizers
type std_weights 't =  ([][]t, []t)
type apply_grad 't  = std_weights t -> std_weights t -> std_weights t

--- Function pairs
--- Denotes a function and it's derivative
type f_pair_1d 't = ([]t -> []t, []t-> []t)
type f_pair_2d 't = ([][]t -> [][]t, [][]t-> [][]t)
