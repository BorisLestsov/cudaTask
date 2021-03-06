#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
       if (abort) exit(code);
   }
}


__global__
void jac_comp(double* a, double* b, int mm, int nn, int kk, int BLOCKSIZE);

__global__
void jac_diff(double* res, double* a, double* b, int mm, int nn, int kk, int BLOCKSIZE);

__global__
void fill(double* a, double value, int mm, int nn, int kk, int BLOCKSIZE);
__global__ 
void jac_max(double* d_array, float* d_max, int elements);
