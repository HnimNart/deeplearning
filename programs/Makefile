DATAFILES=batch_16_mnist_100000_f32.bindata batch_32_mnist_100000_f32.bindata batch_64_mnist_100000_f32.bindata batch_128_mnist_100000_f32.bindata

all: $(DATAFILES)

mnist_100000_f32.bindata:
	wget http://napoleon.hiperfit.dk/~HnimNart/mnist_data/mnist_100000_f32.bindata

batch_%_mnist_100000_f32.bindata: mnist_100000_f32.bindata
	(echo $*; cat $<) > $@

clean:
	rm -f $(DATAFILES) mnist_100000_f32.bindata
