#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "BF_passwords.hpp"

//=============================================================
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

__device__ void my_strcpy(char *dest, const char *src) {
	int i = 0;
	do {
		dest[i] = src[i];
	} while (src[++i] != '\0');
}
__device__ int my_strlen(char *string) {
	int cnt = 0;
	while (string[cnt] != '\0') {
		++cnt;
	}
	return cnt;
}

__device__ int my_comp(char* str1, char* str2, int N) {
	int flag = 0;

	for (int i = 0; i<N; i++) {
		if (str1[i] != str2[i]) {
			flag = 1;
			break;
		}
	}

	return flag;
}

__global__ void bruteforce(char* pass, char* alphabet, char* dest, int N, long long int next) { // N = alphabet length 
	extern __shared__ char s_alphabet[];

	char test[100]; // char test = (char*)malloc(sizeof(char)*N);
	int digit[7] = { 0, };
	int passLen = my_strlen(pass);

	for (int i = 0; i<N; i++)
		s_alphabet[i] = alphabet[i];

	digit[6] = blockIdx.x >= N*N*N ? (int)((blockIdx.x / (N*N*N)) % N) : 0;
	digit[5] = blockIdx.x >= N*N ? (int)((blockIdx.x / (N*N)) % N) : 0;
	digit[4] = blockIdx.x >= N ? (int)((blockIdx.x / N) % N) : 0;
	digit[3] = (int)(blockIdx.x % N);
	digit[2] = threadIdx.x;
	digit[1] = 0;
	digit[0] = 0;

	while (digit[1] < N) {
		for (int i = 0; digit[0] < N; digit[0]++, ++i) {
			test[0] = s_alphabet[digit[0]];

			for (int j = 1; j < passLen; j++) {
				test[j] = s_alphabet[digit[j]];
			}
			test[passLen] = '\0';

			if (!my_comp(pass, test, passLen)) {
				my_strcpy(dest, test);
				dest[passLen] = '\0';
				return;
			}
		}
		++digit[1];
		digit[0] = 0;
	}
}
//=============================================================
#define MAX_PASSWORD_LEN 8

void password_crack();

int main() {

    password_crack();
    return 0;
}

void password_crack() {

    CharSet passwordBF;
    cudaEvent_t start, stop;
    std::ofstream logFile;

    int host_AllCharLen = sizeof(passwordBF.allChar) / sizeof(passwordBF.allChar[0]);
    char* device_Password;
    char* device_allCharacters;
    bool device_checkFind;
    int progressCount=0;
    float workTime;

    logFile.open("bruteForce.log");
    passwordBF.make_password();

    CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMalloc((void**)&device_allCharacters, sizeof(char) * host_AllCharLen + 1));
	CUDA_CHECK(cudaMemcpy(device_allCharacters, passwordBF.allChar, sizeof(char) * host_AllCharLen + 1, cudaMemcpyHostToDevice));

    for(auto password : passwordBF.passwordsHash) {

        std::cout << ++progressCount << "th password(" << password << ") on progress..." << std::endl;

        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMalloc((void**)&device_Password, sizeof(char) * password.length() + 1));
        CUDA_CHECK(cudaMemcpy(device_Password, password.c_str(), sizeof(char) * password.length() + 1, cudaMemcpyHostToDevice));

        dim3 threadsPerBlock(host_AllCharLen, 1);
		dim3 blocksPerGrid((int)std::pow((float)host_AllCharLen, (float)(password.length() - 3)), 1);

        bruteforce<<<blocksPerGrid, threadsPerBlock, sizeof(char) * host_AllCharLen >>>(device_Password, device_allCharacters, device_checkFind, host_AllCharLen, 0);

    	CUDA_CHECK(cudaEventRecord(stop));
		cudaEventSynchronize(stop);

		std::cout << progressCount << "th password completed!" << std::endl;
		CUDA_CHECK(cudaEventElapsedTime(&workTime, start, stop));
		logFile << progressCount << "th password: " << password << "(time consumed=" << workTime << "ms" << ")" << std::endl;

        CUDA_CHECK(cudaFree(device_Password));
    }

    CUDA_CHECK(cudaFree(device_allCharacters));
    logFile.close();
}