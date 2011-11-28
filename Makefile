CUDA_INSTALL_PATH ?=/usr/local/cuda
INCLUDES := -I. -I${CUDA_INSTALL_PATH}/include `pkg-config --cflags glib-2.0`
LIB := -L${CUDA_INSTALL_PATH}/lib64  -lcudart `pkg-config --libs glib-2.0`  
CUDACC = nvcc
CC=gcc
CPP=g++
OBJS=cuda_mem_analysis.o backtrace-symbols.o
#DEBUG = -DCMA_DEBUG

MEMTYPE=-DGPU_MEM_USAGE
#MEMTYPE=-DCPU_MEM_USAGE
CFLAGS= -g ${MEMTYPE} ${INCLUDES} -fPIC ${DEBUG}




default: libcma.so  test
libcma.so: cuda_mem_analysis.o
	${CC}  -shared -o $@ -ldl $< ${LIB} 
libbacktrace-symbols.so: backtrace-symbols.o
	${CC} -shared -o $@  $< -lbfd
test: test.cu
	${CUDACC} -g  ${INCLUDES} -o $@ $< ${LIB}
.c.o:
	${CC} ${CFLAGS} -c -o $@ $<
.cpp.o:
	${CPP} ${CFLAGS} -c -o $@ $<

clean:
	rm -f *.o libcma.so test
deepclean: clean
	rm -f *~
