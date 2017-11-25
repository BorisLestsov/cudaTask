#include <cuda_runtime.h>

__global__
void do_none(){
	int tx = threadIdx.x;
	return;
}

