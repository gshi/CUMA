
#include <execinfo.h>
#include <cxxabi.h>
#include "demangle.h"
#include <stdio.h>
#include <string>


char* 
demangle(char* symbol) {
  int status=0;
  char* demangled;
  demangled = abi::__cxa_demangle(symbol, NULL, NULL, &status);
  if(demangled != NULL){
    //demangled name should be shorter than mangled ones
    //but lets check
    if(strlen(symbol) < strlen(demangled)){
      printf("ERROR: mangled name(%s) is shorter than demangled name(%s)\n",
	     symbol, demangled);
      exit(1);
    }
    int len =strlen(demangled);
    strncpy(symbol, demangled, len);
    //the demangled function name automically added "()", lets remove them
    //by reduce 2 in length
    symbol[len-2]=0;
    free(demangled);
  }else{
    //demangled failed, probably C symbols, leave it intact
  }
  
  return symbol;
}
