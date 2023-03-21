#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "BF_passwords.hpp"

#define MAX_PASSWORD_LEN 8
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
//=============================================================

__device__ int comp_gen(char* str1, char* str2) {
	int flag = 0;

	for (int i = 0; ((str1[i] != '\0') && (str2[i] != '\0')); i++) {
		if ((str1[i] != str2[i])) {
			flag = 1;
			break;
		}
	}

	return flag;
}

__global__ void bruteforce_4_and_5(char* _password, char* _allChar, char* _found, int _allCharLen, long long int next);
__global__ void bruteforce_6(char* _password, char* _allChar, char* _found, int _allCharLen, long long int next);
__global__ void bruteforce_7(char* _password, char* _allChar, char* _found, int _allCharLen, long long int next);
__global__ void bruteforce_8(char* _password, char* _allChar, char* _found, int _allCharLen, long long int next);
void password_crack();

int main() {

    password_crack();
    return 0;
}

void password_crack() {

    CharSet passwordBF;
    cudaEvent_t start, stop;
    std::ofstream logFile;
    std::string result;

    int host_AllCharLen = sizeof(passwordBF.allChar) / sizeof(passwordBF.allChar[0]);
    char* device_Password;
    char* device_allCharacters;
    char* device_found;
    char host_found[8] = {0, };
    int progressCount=0;
    float workTime = 0;

    logFile.open("bruteForce.log");
    passwordBF.make_password();

    CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMalloc((void**)&device_allCharacters, sizeof(char) * host_AllCharLen + 1));
    CUDA_CHECK(cudaMalloc((void**)&device_found, sizeof(char) * MAX_PASSWORD_LEN + 1));
	CUDA_CHECK(cudaMemcpy(device_allCharacters, passwordBF.allChar, sizeof(char) * host_AllCharLen + 1, cudaMemcpyHostToDevice));

    for(auto password : passwordBF.passwordsHash) {

        std::cout << ++progressCount << "th password(" << password << ") on progress..." << std::endl;

        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMalloc((void**)&device_Password, sizeof(char) * password.length() + 1));
        CUDA_CHECK(cudaMemcpy(device_Password, password.c_str(), sizeof(char) * password.length() + 1, cudaMemcpyHostToDevice));

        dim3 threadsPerBlock(host_AllCharLen, 1);
	    dim3 blocksPerGrid((int)std::pow((float)host_AllCharLen, (float)(4)), 1);

        bruteforce_4_and_5<<<blocksPerGrid, threadsPerBlock, sizeof(char) * host_AllCharLen >>>(device_Password, device_allCharacters, device_found, host_AllCharLen, 0);
        CUDA_CHECK(cudaMemcpy(host_found, device_found, sizeof(char) * MAX_PASSWORD_LEN + 1, cudaMemcpyDeviceToHost));

		result = host_found;

		if(result.compare(password) != 0) {
			bruteforce_6<<<blocksPerGrid, threadsPerBlock, sizeof(char) * host_AllCharLen >>>(device_Password, device_allCharacters, device_found, host_AllCharLen, 0);
			CUDA_CHECK(cudaMemcpy(host_found, device_found, sizeof(char) * MAX_PASSWORD_LEN + 1, cudaMemcpyDeviceToHost));
		}

		std::cout << "result: " << result << std::endl;
        if(result.compare(password) == 0) {
            CUDA_CHECK(cudaEventRecord(stop));
		    cudaEventSynchronize(stop);

		    CUDA_CHECK(cudaEventElapsedTime(&workTime, start, stop));
		    logFile << progressCount << "th password: " << result << "(time consumed=" << workTime << "ms" << ")" << std::endl;
        }

        CUDA_CHECK(cudaFree(device_Password));
    }

    CUDA_CHECK(cudaFree(device_allCharacters));
    logFile.close();
}

// 63^3 = 250047
// 63^4 = 15752961
// 94^3 = 830584
// 94^4 = 78074896
// 512 * 512 * 64 = 16777216
// 2^32 - 1 = 4.3 * 10^9
// 94^5 = 7.3 * 10^9

__global__ void bruteforce_4_and_5(char* _password, char* _allChar, char* _found, int _allCharLen, long long int next) {
	
	extern __shared__ char s_alphabet[];

	char candidate[MAX_PASSWORD_LEN + 1]; // char candidate = (char*)malloc(sizeof(char)*N);
	int digit[8] = { 0, };
	//int passLen = my_strlen(_password);

	for (int i = 0; i < _allCharLen; i++)
		s_alphabet[i] = _allChar[i];

	digit[4] = blockIdx.x >= _allCharLen * _allCharLen * _allCharLen ? (int)((blockIdx.x / (_allCharLen * _allCharLen * _allCharLen)) % _allCharLen) : 0;
	digit[3] = blockIdx.x >= _allCharLen * _allCharLen ? (int)((blockIdx.x / (_allCharLen * _allCharLen)) % _allCharLen) : 0;
	digit[2] = blockIdx.x >= _allCharLen ? (int)((blockIdx.x / _allCharLen) % _allCharLen) : 0;
	digit[1] = (int)(blockIdx.x % _allCharLen);
	digit[0] = threadIdx.x;

	//=====================Search for 4, 5 characters=====================//
	// 4 characters
	for(int j = 0; j < 4; j++) {
		candidate[j] = s_alphabet[digit[j]];
	}
        candidate[4] = '\0';

    if(!comp_gen(_password, candidate)) {
		my_strcpy(_found, candidate);
		_found[4] = '\0';
		return;
	}

	// 5 characters
	for(int j = 0; j < 5; j++) {
		candidate[j] = s_alphabet[digit[j]];
	}
        candidate[5] = '\0';

    if(!comp_gen(_password, candidate)) {
		my_strcpy(_found, candidate);
		_found[5] = '\0';
		return;
	}
	//=====================END Search for 4, 5 characters=====================//
}

__global__ void bruteforce_6(char* _password, char* _allChar, char* _found, int _allCharLen, long long int next) {
	
	extern __shared__ char s_alphabet[];

	char candidate[MAX_PASSWORD_LEN + 1]; // char candidate = (char*)malloc(sizeof(char)*N);
	int digit[8] = { 0, };
	//int passLen = my_strlen(_password);

	for (int i = 0; i < _allCharLen; i++)
		s_alphabet[i] = _allChar[i];

	digit[5] = 0;
	digit[4] = blockIdx.x >= _allCharLen * _allCharLen * _allCharLen ? (int)((blockIdx.x / (_allCharLen * _allCharLen * _allCharLen)) % _allCharLen) : 0;
	digit[3] = blockIdx.x >= _allCharLen * _allCharLen ? (int)((blockIdx.x / (_allCharLen * _allCharLen)) % _allCharLen) : 0;
	digit[2] = blockIdx.x >= _allCharLen ? (int)((blockIdx.x / _allCharLen) % _allCharLen) : 0;
	digit[1] = (int)(blockIdx.x % _allCharLen);
	digit[0] = threadIdx.x;

	// 6 characters
	for(; digit[5] < _allCharLen; digit[5]++) {
		for (int j = 0; j < 5; j++) {
			candidate[j] = s_alphabet[digit[j]];
		}
		candidate[5] = s_alphabet[digit[5]];

        candidate[6] = '\0';

    	if (!comp_gen(_password, candidate)) {
			my_strcpy(_found, candidate);
			_found[6] = '\0';
			return;
		}
	}
}