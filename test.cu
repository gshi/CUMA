#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#define N 16

__global__ void vecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
	C[i] = A[i] + B[i];
}

int testfunc()
{
    int i;
    float* A;
    float* B;
    float* C;
    float* hostA;
    float* hostB;
    float* hostC;
    
    
    hostA = (float*) malloc(sizeof(float)*N);
    hostB = (float*) malloc(sizeof(float)*N);
    hostC = (float*) malloc(sizeof(float)*N);

    for (i=0;i < N;i++){
	hostA[i] = i+1;
	hostB[i] = i+101;
	hostC[i] = 0;
    }
 
 
    cudaMalloc((void**)&A, sizeof(float)*N);
    cudaMalloc((void**)&B, sizeof(float)*N);
    cudaMalloc((void**)&C, sizeof(float)*N);
    cudaMemset(A, 0, sizeof(float)*N);	
    cudaMemset(B, 0, sizeof(float)*N);	
    cudaMemset(C, 0, sizeof(float)*N);	
    cudaMemcpy(A, hostA, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(B, hostB, sizeof(float)*N, cudaMemcpyHostToDevice);

    //Kernel invocation
    vecAdd<<<1, N>>>(A, B, C);

    cudaMemcpy(hostC, C, sizeof(float)*N, cudaMemcpyDeviceToHost);
    
    cudaThreadSynchronize();
	
    for (i =0;i < N; i++){
	printf("C[%d]=%f\n", i, hostC[i]);

    }	
    for (i=0;i < N ;i++){
	if (hostC[i] != (hostA[i]+hostB[i])){
	    printf("ERROR: data not match, exitting\n");
	    exit(1);
	}
    }
    
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    printf("hello world runs sucessfully in GPU!\n");

    free(hostA);
    free(hostB);
    //free(hostC);

    return 0;
    
}

int main()
{
	testfunc();
	
	return 0;
}
