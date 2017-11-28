#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#include "cuda_runtime.h"
#include "kernels.cuh"

#include "mpi.h"

#define  Max(a,b) ((a)>(b)?(a):(b))

FILE *in;
int TRACE = 1;
int i, j, k, it;
double EPS;
int     M, N, K, ITMAX;
double  MAXEPS = 0.1;
double time0;

double *A;
#define A(i,j,k) A[((i)*N+(j))*K+(k)]

double solution(int i, int j, int k)
{
    double x = 10.*i / (M - 1), y = 10.*j / (N - 1), z = 10.*k / (K - 1);
    return 2.*x*x - y*y - z*z;
    /*    return x+y+z; */
}

double jac(double *a, int mm, int nn, int kk, int itmax, double maxeps);

int main(int an, char **as)
{

    in = fopen("data3.in", "r");
    if (in == NULL) { printf("Can not open 'data3.in' "); exit(1); }
    i = fscanf(in, "%d %d %d %d %d", &M, &N, &K, &ITMAX, &TRACE);
    if (i < 4) 
    {
        printf("Wrong 'data3.in' (M N K ITMAX TRACE)");
        exit(2);
    }

    A = (double*) malloc(M*N*K*sizeof(double));

    for (i = 0; i <= M - 1; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= K - 1; k++)
            {
                if (i == 0 || i == M - 1 || j == 0 || j == N - 1 || k == 0 || k == K - 1)
                    A(i, j, k) = solution(i, j, k);
                else 
                    A(i, j, k) = 0.;
            }


    printf("%dx%dx%d x %d\t<", M, N, K, ITMAX);
    time0 = 0.;
    EPS = jac(A, M, N, K, ITMAX, MAXEPS);   
    
    printf("%3.1f>\teps=%.4g ", time0, EPS);

    if (TRACE)
    {
        EPS = 0.;

        for (i = 0; i <= M - 1; i++)
            for (j = 0; j <= N - 1; j++)
                for (k = 0; k <= K - 1; k++)
                    EPS = Max(fabs(A(i, j, k) - solution(i, j, k)), EPS);
        printf("delta=%.4g\n", EPS);
    }

    free(A);
    return 0;
}

#define a(i,j,k) a[((i)*nn+(j))*kk+(k)]
#define b(i,j,k) b[((i)*nn+(j))*kk+(k)]

#define BLOCKSIZE 8
#define REDUCE_THREADS 128

double jac(double *a, int mm, int nn, int kk, int itmax, double maxeps)
{
    gpuErrchk(cudaSetDevice(0));
    
    double *b, *b_d, *a_d, *d_buf;
    double eps;

    float flt_min = FLT_MIN;
    float* d_eps;
    gpuErrchk(cudaMalloc(&d_eps, sizeof(float)));


    b = (double*) malloc(mm*nn*kk*sizeof(double));

    gpuErrchk(cudaMalloc(&b_d, mm*nn*kk*sizeof(double)));
    gpuErrchk(cudaMalloc(&a_d, mm*nn*kk*sizeof(double)));
    gpuErrchk(cudaMalloc(&d_buf, mm*nn*kk*sizeof(double)));

    int mm_dim = (mm-2)/BLOCKSIZE + ((mm-2)%BLOCKSIZE!=0);
    int nn_dim = (nn-2)/BLOCKSIZE + ((nn-2)%BLOCKSIZE!=0);
    int kk_dim = (kk-2)/BLOCKSIZE + ((kk-2)%BLOCKSIZE!=0);
    dim3 blockGrid  = dim3(mm_dim, nn_dim, kk_dim);
    dim3 threadGrid = dim3(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);

    int mm_dim1 = (mm)/BLOCKSIZE + ((mm)%BLOCKSIZE!=0);
    int nn_dim1 = (nn)/BLOCKSIZE + ((nn)%BLOCKSIZE!=0);
    int kk_dim1 = (kk)/BLOCKSIZE + ((kk)%BLOCKSIZE!=0);
    dim3 blockGrid1  = dim3(mm_dim1, nn_dim1, kk_dim1);
    dim3 threadGrid1 = dim3(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);


    gpuErrchk(cudaMemcpy(a_d, a, mm*nn*kk*sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(b_d, a_d, mm*nn*kk*sizeof(double), cudaMemcpyDeviceToDevice));
    fill<<<blockGrid1, threadGrid1>>>(d_buf, -DBL_MAX, mm, nn, kk, BLOCKSIZE);    
    
    for (it = 1; it <= itmax - 1; it++)
    {
        jac_comp<<<blockGrid, threadGrid>>>(a_d, b_d, mm, nn, kk, BLOCKSIZE);
        //gpuErrchk(cudaMemcpy(b, b_d, mm*nn*kk*sizeof(double), cudaMemcpyDeviceToHost));
        
        
        jac_diff<<<blockGrid, threadGrid>>>(d_buf, a_d, b_d, mm, nn, kk, BLOCKSIZE);
        
        //for (i = 0; i < mm; ++i)
        //for (j = 0; j < nn; ++j)
        //for (k = 0; k < kk; ++k)
        //printf("%d %d %d %f \n", i, j, k, b(i,j,k));

        float epsf;
        gpuErrchk(cudaMemcpy(d_eps, &flt_min, sizeof(float), cudaMemcpyHostToDevice));
        int blocks = (mm*nn*kk)/REDUCE_THREADS + ((mm*nn*kk)%REDUCE_THREADS!=0);
        max_reduce<<<blocks, REDUCE_THREADS, REDUCE_THREADS*sizeof(double)>>>(d_buf, d_eps, mm*nn*kk);
        
        gpuErrchk(cudaMemcpy(&epsf, d_eps, sizeof(float), cudaMemcpyDeviceToHost));
        eps = (double) epsf;        
        
        gpuErrchk(cudaMemcpy(a_d, b_d, mm*nn*kk*sizeof(double), cudaMemcpyDeviceToDevice));
        //gpuErrchk(cudaGetLastError());
        if (TRACE && it%TRACE == 0)
            printf("\nIT=%d eps=%.4g\t", it, eps);
        if (eps < maxeps) 
            break;
    }
    free(b);
    return eps;
}
