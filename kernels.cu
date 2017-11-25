#include <cuda_runtime.h>

__global__
void do_none(double* A, double* B){
	int tx = threadIdx.x;
	A[tx] = B[tx];
	return;
}

