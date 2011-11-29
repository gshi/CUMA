#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
 #include <dlfcn.h>

#define N  16

#define CUERR do {  \
    if (err != CUDA_SUCCESS){			\
      printf("ERROR: CU call failed (%s) at file %s, line %d\n",\
	     cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__); \
      exit(1);								\
    }						\
  }while(0)



int (*__cuInit)(int);
int (*__cuDeviceGetCount)(int*);
int (*__cuDriverGetVersion)(int*);
int (*__cuDeviceGet)(int*, int);
int (*__cuDeviceGetAttribute)(int*, int, int);
int (*__cuDeviceGetName)(char*, int, int);
int (*__cuDeviceTotalMem)(unsigned int*, int);
int (*__cuDeviceComputeCapability)(int*, int*, int);
int (*__cuCtxCreate)(CUcontext*, unsigned int, unsigned int);
int (*__cuCtxDestroy)(void*);
int (*__cuMemAlloc)(unsigned int*, unsigned int);
int (*__cuMemFree)(unsigned int);
int (*__cuMemGetInfo)(unsigned int*, unsigned int*);
#define cudalib RTLD_DEFAULT

int testfunc()
{
    CUdeviceptr A = 0;
    CUdeviceptr B = 0;
    CUdeviceptr C = 0;
    CUresult err;
    CUdevice device;
    CUcontext context;
    int devid = 0;

    CUresult (*real_cuMemAlloc)(CUdeviceptr* dptr, size_t bytesize);

    __cuDeviceGetCount = (int(*)(int*)) dlsym(cudalib, "cuDeviceGetCount");
    __cuDriverGetVersion = (int(*)(int*)) dlsym( cudalib, "cuDriverGetVersion" );
    __cuInit = (int(*)(int)) dlsym( cudalib, "cuInit" );
    __cuDeviceGet = (int(*)(int*, int)) dlsym( cudalib, "cuDeviceGet" );
    __cuDeviceGetAttribute = (int(*)(int*, int, int)) dlsym( cudalib, "cuDeviceGetAttribute" );
    __cuDeviceGetName = (int(*)(char*, int, int)) dlsym( cudalib, "cuDeviceGetName" );
    __cuDeviceTotalMem = (int(*)(unsigned int*, int)) dlsym( cudalib, "cuDeviceTotalMem" );
    __cuDeviceComputeCapability = (int(*)(int*, int*, int)) dlsym( cudalib, "cuDeviceComputeCapability" );
    __cuCtxCreate = (int(*)(CUcontext*, unsigned int, unsigned int)) dlsym( cudalib, "cuCtxCreate" );
    __cuCtxDestroy = (int(*)(void*)) dlsym( cudalib, "cuCtxDestroy" );
    __cuMemAlloc = (int(*)(unsigned int*, unsigned int)) dlsym( cudalib, "cuMemAlloc" );
    __cuMemFree = (int(*)(unsigned int)) dlsym( cudalib, "cuMemFree" );
    __cuMemGetInfo = (int(*)(unsigned int*, unsigned int*)) dlsym( cudalib, "cuMemGetInfo" );




    
    //cuInit(0);
    //err = cuDeviceGet(&device, devid); CUERR;
    //err = cuCtxCreate(&context, 0, device); CUERR;

    (*__cuInit)(0);
    err = (CUresult)(*__cuDeviceGet)(&device, devid); CUERR;
    err = (CUresult)(*__cuCtxCreate)(&context, 0, device); CUERR;


    real_cuMemAlloc = (CUresult (*)(CUdeviceptr* dptr, size_t bytesize))
      dlsym(RTLD_NEXT, "cuMemAlloc");
    printf("real_cuMemAlloc=%p\n", real_cuMemAlloc);
    err = (*real_cuMemAlloc)(&A, sizeof(float)*N); //CUERR;
    printf("err=%d\n", err);
    printf("A=%p\n", A);
    
    //err = cuMemAlloc(&A, sizeof(float)*N); CUERR;
    err = cuMemAlloc(&B, sizeof(float)*N); //CUERR;
    //err = cuMemAlloc(&C, sizeof(float)*N); CUERR;
    
    printf("B=%p, err=%d\n", B,err);
    printf("C=%p\n", C);
    
    //cuMemFree(A);
    //cuMemFree(B);
    //cuMemFree(C);
    
    //err = cuCtxDetach(context); CUERR;
    return 0;
    
}

int main()
{
  testfunc();
  
  return 0;
}
