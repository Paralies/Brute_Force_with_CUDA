#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#if defined(NDEBUG)
#define CUDA_CHECK(x)	(x)
#else
#define CUDA_CHECK(x)	do {\
		(x); \
		cudaError_t e = cudaGetLastError(); \
		if (cudaSuccess != e) { \
			printf("cuda failure \"%s\" at %s:%d\n", \
			       cudaGetErrorString(e), \
			       __FILE__, __LINE__); \
			exit(1); \
		} \
	} while (0)
#endif

int main() {

    std::ofstream logFile;
    logFile.open("bruteForce.log");

    std::ofstream passwordFile;
    passwordFile.open("password.txt");

    CharSet passwordBF;
    passwordBF.make_password();

    for(auto i : passwordBF.passwordsHash) {
        passwordFile << i << std::endl;
        //std::cout << i << std::endl;
    }
    
    logFile.close();
    passwordFile.close();
}