CUDA_INSTALL_PATH ?=/usr/local/cuda
INCLUDES := -I. -I${CUDA_INSTALL_PATH}/include `pkg-config --cflags glib-2.0`
LIB := -L${CUDA_INSTALL_PATH}/lib64  -lcudart `pkg-config --libs glib-2.0`  
CUDACC = nvcc
CC=gcc
CPP=g++
OBJS=cuda_mem_analysis.o demangle.o
#DEBUG = -DCMA_DEBUG

MEMTYPE=-DGPU_MEM_USAGE
#MEMTYPE=-DCPU_MEM_USAGE
CFLAGS= -g ${MEMTYPE} ${INCLUDES} -fPIC ${DEBUG}
LD=gcc



default: libcma.so  test
all: libcma.so  test
libcma.so: ${OBJS}
	${CC}  -shared -o $@ -ldl ${OBJS} ${LIB} -lbfd -liberty
test: test.cu
	${CUDACC} -g -c  ${INCLUDES} $<
	${LD}  -o $@ test.o  ${LIB}
.c.o:
	${CC} ${CFLAGS} -c -o $@ $<
.cpp.o:
	${CPP} ${CFLAGS} -c -o $@ $<

clean:
	rm -f *.o libcma.so test
deepclean: clean
	rm -f *~
