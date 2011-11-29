#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#define N  16

int testfunc()
{
    float* A;
    float* B;
    float* C;
    cudaMalloc((void**)&A, sizeof(float)*N);
    cudaMalloc((void**)&B, sizeof(float)*N);
    cudaMalloc((void**)&C, sizeof(float)*N);

    //cudaFree(A);
    //cudaFree(B);
    cudaFree(C);

    return 0;
    
}

int main()
{
	testfunc();
	
	return 0;
}
