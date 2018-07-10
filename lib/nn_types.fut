type forwards 'input 'w 'output 'garbage  = w -> input -> (garbage, output) --- NN
type backwards 'g 'w  'err1  'err2  = w -> g ->  err1  -> (err2, w)
type update 'w '^f = f ->  w -> w -> w

type updater 'w = w -> w -> w


type NN 'input 'w 'output 'g 'e1 'e2  '^f = (forwards input w output g,
                                         backwards g w e1 e2 ,
                                         update w f,
                                         w)
