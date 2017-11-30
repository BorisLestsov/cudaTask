#include <cuda_runtime.h>
#include <float.h>


__global__
void solution_ker(double* A, int M, int N, int K){
   int i = threadIdx.x;
   int j = threadIdx.y;
   int k = threadIdx.z;
   int ind = (i*N+j)*K + k; 
   if (i == 0 || j == M-1 || j == 0 || j == N-1 || k == 0 | k == K-1)   {
       A[ind] = 10.*i ; //finish this
   } else {
       A[ind] = 0.0;
   }
}

#define a(i,j,k) a[((i)*nn+(j))*kk+(k)]
#define b(i,j,k) b[((i)*nn+(j))*kk+(k)]
#define res(i,j,k) res[((i)*nn+(j))*kk+(k)]

__global__
void jac_comp(double* a, double* b, int mm, int nn, int kk, int BLOCKSIZE){
    int bl_i = blockIdx.x;
    int bl_j = blockIdx.y;
    int bl_k = blockIdx.z;
    int th_i = threadIdx.x;
    int th_j = threadIdx.y;
    int th_k = threadIdx.z;
    int i = BLOCKSIZE*bl_i + th_i + 1;
    int j = BLOCKSIZE*bl_j + th_j + 1;
    int k = BLOCKSIZE*bl_k + th_k + 1;
    if (i >= mm-1 || j >= nn-1 || k >= kk-1)
        return;

    b(i,j,k) = (a(i-1, j ,k) + a(i+1, j, k)
             +  a(i, j-1, k) + a(i, j+1, k)
             +  a(i, j, k-1) + a(i, j, k+1)) / 6.;
    //b(i,j,k) = 1.0;
}

__global__
void no_bound_memcpy(double* a, double* b, int mm, int nn, int kk, int BLOCKSIZE, double value){
    int bl_i = blockIdx.x;
    int bl_j = blockIdx.y;
    int bl_k = blockIdx.z;
    int th_i = threadIdx.x;
    int th_j = threadIdx.y;
    int th_k = threadIdx.z;
    int i = BLOCKSIZE*bl_i + th_i + 1;
    int j = BLOCKSIZE*bl_j + th_j + 1;
    int k = BLOCKSIZE*bl_k + th_k + 1;
    if (i >= mm-1 || j >= nn-1 || k >= kk-1)
        return;

    a(i,j,k) = b(i,j,k);
}

__global__
void fill(double* a, double value, int mm, int nn, int kk, int BLOCKSIZE){
    int bl_i = blockIdx.x;
    int bl_j = blockIdx.y;
    int bl_k = blockIdx.z;
    int th_i = threadIdx.x;
    int th_j = threadIdx.y;
    int th_k = threadIdx.z;
    int i = BLOCKSIZE*bl_i + th_i;
    int j = BLOCKSIZE*bl_j + th_j;
    int k = BLOCKSIZE*bl_k + th_k;

    a(i,j,k) = value;
}


__global__
void jac_diff(double* res, double* a, double* b, int mm, int nn, int kk, int BLOCKSIZE){
    int bl_i = blockIdx.x;
    int bl_j = blockIdx.y;
    int bl_k = blockIdx.z;
    int th_i = threadIdx.x;
    int th_j = threadIdx.y;
    int th_k = threadIdx.z;
    int i = BLOCKSIZE*bl_i + th_i + 1;
    int j = BLOCKSIZE*bl_j + th_j + 1;
    int k = BLOCKSIZE*bl_k + th_k + 1;
    if (i >= mm-1 || j >= nn-1 || k >= kk-1)
        return;

    res(i,j,k) = fabs(a(i,j,k)-b(i,j,k));
}


__device__
float atomicMaxf(float* address, float val)
{
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                __float_as_int(val));
    }
    return __int_as_float(old);
}


__global__
void jac_max(double* d_array, float* d_max, int elements)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;
    shared[tid] = -FLT_MAX; 

    while (gid < elements) {
        shared[tid] = max(shared[tid], d_array[gid]);
        gid += gridDim.x*blockDim.x;
    }
    __syncthreads();
    gid = (blockDim.x * blockIdx.x) + tid;
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s && gid < elements)
            shared[tid] = max(shared[tid], shared[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        atomicMaxf(d_max, __double2float_rn(shared[0]));
}


__global__
void max_reduce_no_bounds(double* a, int mm, int nn, int kk, int BLOCKSIZE){
    

}

