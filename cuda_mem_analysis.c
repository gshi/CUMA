#define __USE_GNU
#define _GNU_SOURCE

#include "backtrace-symbols.c"

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>
#include <glib.h>
#include <string.h>
#include <execinfo.h>
#include <dlfcn.h>
#include "demangle.h"

#ifdef CMA_DEBUG
#define FUNC_ENTER_LOG do {						\
    printf("Entering %s\n", __FUNCTION__);				\
  }while(0)

#define FUNC_EXIT_LOG do {						\
    printf("Exitting %s\n", __FUNCTION__);				\
  }while(0)
#else
#define FUNC_ENTER_LOG 
#define FUNC_EXIT_LOG
#endif

#if 1

#define PRINTF(...) do {                            \
    printf(__VA_ARGS__);			    \
    fflush(stdout); } while (0)       

#else
#define PRINTF(...)
#endif		

#define MAX_NAME_LEN 128
#define MAX_NUM_STACKS 16
typedef struct mem_region_s{
  size_t size;
  void* ptr;
  int num_stacks;
  char backtrace[MAX_NUM_STACKS][MAX_NAME_LEN]; //the calling function's name
}mem_region_t;



#ifdef GPU_MEM_USAGE
char memtype[] = "GPU";
#else //CPU_MEM_USAGE
char memtype[] = "CPU";
#endif

extern const char *__progname;
extern const char *__progname_full;


static cudaError_t (*real_cudaMalloc)(void ** devPtr, size_t size);
static cudaError_t (*real_cudaFree)(void* devPtr);
static void* (*real_malloc)(size_t size);
static void (*real_free)(void* ptr);
static void* handle = NULL;
static GHashTable* hash_table = NULL;

static int entry_idx =0;
struct mem_stat{
  size_t tot_mem_alloc;
  size_t peak_mem_alloc;
  int num_alloc;
  int num_free;
}mem_stat;

int cma_init()
{
  FUNC_ENTER_LOG;
  static int cma_initialized = 0;
  if(cma_initialized ){
    goto out;
  }
  cma_initialized =1;
  
  handle  = dlopen("libcudart.so", RTLD_NOW);
  if(handle == NULL){
    printf("ERROR: opening libcudart.so failed\n");
    exit(1);
  }
  
  real_cudaMalloc = dlsym(handle, "cudaMalloc");
  real_cudaFree = dlsym(handle, "cudaFree");
  if(real_cudaMalloc == NULL || real_cudaFree == NULL){
    printf("ERROR: symbol cudaMalloc/cudaFree not found\n");
    exit(1);
  }
  
  hash_table = g_hash_table_new(g_direct_hash, g_direct_equal);
  
  
  mem_stat.tot_mem_alloc = 0;
  mem_stat.num_alloc = 0;
  mem_stat.num_free = 0;
  
  
 out:
  FUNC_EXIT_LOG;
  
  return 0;
}

int cpu_cma_init()
{
  FUNC_ENTER_LOG;
  static int cma_initialized = 0;
  if(cma_initialized ){
    goto out;
  }
  cma_initialized =1;
  
  real_malloc = dlsym(RTLD_NEXT, "malloc");
  real_free = dlsym(RTLD_NEXT, "free");
  if(real_malloc == NULL || real_free == NULL){
    printf("ERROR: symbol malloc/free not found, function %s\n", __FUNCTION__);
    exit(1);
  }
  
  hash_table = g_hash_table_new(g_direct_hash, g_direct_equal);  
  
  mem_stat.tot_mem_alloc = 0;
  mem_stat.num_alloc = 0;
  mem_stat.num_free = 0;
  
  
 out:
  FUNC_EXIT_LOG;
  
  return 0;
}

static void
print_one_entry(gpointer key, gpointer value, gpointer user_data)
{
  int i;
  mem_region_t* mr  = (mem_region_t*) value;
  PRINTF("entry %d: ptr=%p, size=%d. The calling backtrace is\n", entry_idx, mr->ptr, mr->size, mr->backtrace);
  for(i=0;i < mr->num_stacks; i++){
    PRINTF("        %s\n", mr->backtrace[i]);
    //PRINTF("        %s\n", demangle(mr->backtrace[i]));
  }
  entry_idx ++;
  return;  
}

void 
cma_print_stat()
{
  PRINTF("Summary of %s memory usage\n", memtype);
  PRINTF("Memory usage: %d alloc, %d free, peak_memory_usage=%lld, current memory in usage: %lld\n",  
	 mem_stat.num_alloc, mem_stat.num_free, 
	 mem_stat.peak_mem_alloc, mem_stat.tot_mem_alloc);
}


void
cma_print_detail()
{
  cma_print_stat();
  PRINTF("There are %d entries in the hash table\n", g_hash_table_size(hash_table));
  
  entry_idx = 0;
  g_hash_table_foreach(hash_table, print_one_entry, NULL);
  
  return;
}

int cma_cleanup()
{
  dlclose(handle);  handle = NULL;
  g_hash_table_destroy(hash_table); hash_table = NULL;

}
static __attribute__((constructor)) void
cma_constructor( void )
{
}

static __attribute__((destructor)) void
cma_destructor(void)
{
  cma_print_detail();
}

/*******************************************************************************************
 ************************* Intercepted Functions ********************************************
********************************************************************************************/
#ifdef GPU_MEM_USAGE
cudaError_t cudaMalloc(void ** devPtr,  size_t size)	
{
  FUNC_ENTER_LOG;
  cma_init();
  
  cudaError_t rc = (*real_cudaMalloc)(devPtr, size);

  //get the stack back trace
  int buflen = MAX_NUM_STACKS ;
  void* buffer[buflen];
  int n = backtrace(buffer, buflen);
  if(n <2){
    printf("Error: backtrace call failed(n=%d)\n",n);
    exit(1);
  }
  
  
  char** s = backtrace_symbols(buffer, n);
  if(s == NULL){
    printf("Error: bactrace_symbols call failed\n");
    exit(1);
  }
  

  mem_region_t * mr = malloc(sizeof(mem_region_t));
  if(mr == NULL){
    printf("ERROR: malloc faied for mem_region_t\n");
    exit(1);
  }
  
  mr->ptr = *devPtr;
  mr->size= size;    
  mr->num_stacks= n;
  int i;
  for(i=0;i < n; i++){
    strncpy(mr->backtrace[i], s[i], MAX_NAME_LEN);
  }

  g_hash_table_insert(hash_table, mr->ptr, mr);
  
  mem_stat.num_alloc ++;
  mem_stat.tot_mem_alloc += mr->size;
  if(mem_stat.tot_mem_alloc > mem_stat.peak_mem_alloc){
    mem_stat.peak_mem_alloc = mem_stat.tot_mem_alloc;
  }

  free(s);
  
  FUNC_EXIT_LOG;
  return rc;
}


cudaError_t cudaFree(void* devPtr)
{
  FUNC_ENTER_LOG;
  cma_init();
 
  mem_region_t* mr = (mem_region_t*)g_hash_table_lookup(hash_table, devPtr);
  if(mr == NULL){
    printf("ERROR: memory region not found in hash table\n");
    exit(1);
  }
  
  mem_stat.num_free ++;
  mem_stat.tot_mem_alloc -= mr->size;

  g_hash_table_remove(hash_table, devPtr);
  free(mr);
  
  cudaError_t rc = (*real_cudaFree)(devPtr);
  
  
  FUNC_EXIT_LOG;

  return rc;
}
#endif

/******************************************************************
 *
 * host: malloc and free
 *
 *********************************************************************/
#ifdef CPU_MEM_USAGE
void* malloc(size_t size)	
{  
  FUNC_ENTER_LOG;
  static int depth = 0;  
  depth ++;

  
  cpu_cma_init();
  
  void* ptr= real_malloc(size);
  if(ptr == NULL){
    printf("Error: real_malloc failed in %s\n", __FUNCTION__);
    depth--;
    exit(1);
  }
  
  if(depth > 1){
    //we are in a loop of ourselves, get out now
    depth--;
    return ptr;
  }
  
  //get the stack back trace
  int buflen = MAX_NUM_STACKS ;
  void* buffer[buflen];
  int n = backtrace(buffer, buflen);
  if(n <2){
    printf("Error: backtrace call failed(n=%d)\n",n);
    depth--;
    exit(1);
  }
  
  
  char** s = backtrace_symbols(buffer, n);
  if(s == NULL){
    printf("Error: bactrace_symbols call failed\n");
    depth--;    
    exit(1);
  }
  
  if(!strncmp(s[1], __progname_full, strlen(__progname_full)) == 0){
    //only record cals from the exeutible, not from libs
    real_free(s);
    depth--;
    return ptr;
  }
  
  mem_region_t * mr = real_malloc(sizeof(mem_region_t));
  if(mr == NULL){
    printf("ERROR: malloc faied for mem_region_t\n");
    depth--;
    exit(1);
  }
  
  mr->ptr = ptr;
  mr->size= size;    
  mr->num_stacks= n;
  int i;
  for(i=0;i < n; i++){
    strncpy(mr->backtrace[i], s[i], MAX_NAME_LEN);
  }

  g_hash_table_insert(hash_table, mr->ptr, mr);
  
  mem_stat.num_alloc ++;
  mem_stat.tot_mem_alloc += mr->size;
  if(mem_stat.tot_mem_alloc > mem_stat.peak_mem_alloc){
    mem_stat.peak_mem_alloc = mem_stat.tot_mem_alloc;
  }
  
  real_free(s);
  
  depth -- ;
  FUNC_EXIT_LOG;
  return ptr;
}


void free(void* devPtr)
{
  FUNC_ENTER_LOG;
  cpu_cma_init();
 
  mem_region_t* mr = (mem_region_t*)g_hash_table_lookup(hash_table, devPtr);
  if(mr == NULL){
    //printf("ERROR: memory region not found in hash table\n");
    //exit(1);
    return;
  }
  
  mem_stat.num_free ++;
  mem_stat.tot_mem_alloc -= mr->size;
  
  g_hash_table_remove(hash_table, devPtr);
  real_free(mr);
  
  (*real_free)(devPtr);  
  
  FUNC_EXIT_LOG;
  
  return;
}


#endif
