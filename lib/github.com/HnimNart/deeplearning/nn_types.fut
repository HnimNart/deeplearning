-- | Network types
type^ forwards   'input 'w 'output 'cache = bool -> w -> input -> (cache, output)
type^ backwards  'c 'w  'err_in  'err_out '^u = bool -> u -> w -> c -> err_in  -> (err_out, w)

type^ NN 'input 'w 'output 'c 'e_in 'e_out '^u =
               { forward : (k: i64) -> forwards ([k]input) w ([k]output) ([k]c),
                 backward: (k: i64) -> backwards ([k]c) w ([k]e_in) ([k]e_out) u,
                 weights : w}

--- Commonly used types
type arr1d [n] 't = [n]t
type arr2d [n][m] 't = [n][m]t
type arr3d [n][m][p] 't = [n][m][p]t
type arr4d [n][m][p][q] 't = [n][m][p][q]t

type dims2d  = (i32, i32)
type dims3d  = (i32, i32, i32)

--- The 'standard' weight definition
--- used by optimizers
type std_weights [a][b][c] 't = ([a][b]t, [c]t)
type^ apply_grad2 'x 'y = (x, y) -> (x, y) -> (x, y)
type^ apply_grad3 't = (a: i64) -> (b: i64) -> apply_grad2 ([a][b]t) ([a]t)


--- Function pairs
--- Denotes a function and it's derivative
type^ activation_func 'o = {f:o -> o, fd:o -> o}
type^ loss_func 'o  't   = {f:o -> o -> t, fd:o -> o -> o}
