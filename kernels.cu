#include <cuda_runtime.h>


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
void jac_eps(double* a, double* b, int mm, int nn, int kk){
   int i = threadIdx.x+1;
   int j = threadIdx.y+1;
   int k = threadIdx.z+1;
   b(i,j,k) = (a(i-1, j ,k) + a(i+1, j, k)
            +  a(i, j-1, k) + a(i, j+1, k)
            +  a(i, j, k-1) + a(i, j, k+1)) / 6.;
}


