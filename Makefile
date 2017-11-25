CFLAGS= -O2
CC= gcc
NVCC= nvcc
CUFLAGS= -I. -O2 -g
LINKFLAGS= -L/usr/local/cuda/lib64 -lcudart

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
