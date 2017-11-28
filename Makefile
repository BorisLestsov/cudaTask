CFLAGS= -O2 -g
CC= g++
NVCC= nvcc
CUFLAGS= -I. -O2 -g -gencode=arch=compute_20,code=compute_20 -I/usr/include/mpich-x86_64/
LINKFLAGS= -L/usr/local/cuda/lib64 -lcudart -L/usr/lib64/mpich-3.2/lib/ -lmpi

.PHONY: target clean

all: target

target: main

main: main.o kernels.o 
	$(CC) $(LINKFLAGS) main.o kernels.o -o main

main.o: main.cu
	$(NVCC) $(CUFLAGS)  main.cu -c -o main.o

kernels.o: kernels.cu
	$(NVCC) $(CUFLAGS) kernels.cu -c -o kernels.o

clean:
	rm *.o
	rm main
