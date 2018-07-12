type forwards 'input 'w 'output 'garbage  = bool -> w -> input -> (garbage, output) --- NN
type backwards 'g 'w  'err1  'err2  = w -> g ->  err1  -> (err2, w)
type update 'w '^f = f -> w -> w -> w

type updater 'w = w -> w -> w


type NN 'input 'w 'output 'g 'e1 'e2  '^f = (forwards input w output g,
                                         backwards g w e1 e2 ,
                                         update w f,
                                         w)


(bool -> w1 -> []i0 -> (g2, []o5), w1 -> g2 -> []o5 -> (e24, w1),
([][]gsd.t, []gsd.t) -> ([][]gsd.t,[]gsd.t) -> ([][]gsd.t,[]gsd.t) -> w1 -> w1 -> w1, w1)

(bool -> w -> []i -> (g, []o), w -> g -> e1 -> (e2, w), ([][]R.t, []R.t) -> ([][]R.t,
                                                                      []R.t) -> ([][]R.t,
                                                                                 []R.t) -> w -> w -> w,
   w