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

__global__ void bruteforce(char* _password, char* _allChar, char* _found, int _allCharLen, long long int next) {
	extern __shared__ char s_alphabet[];

	char candidate[8]; // char candidate = (char*)malloc(sizeof(char)*N);
	int digit[8] = { 0, };
	//int passLen = my_strlen(_password);

	for (int i = 0; i < _allCharLen; i++)
		s_alphabet[i] = _allChar[i];

	digit[7] = blockIdx.x >= _allCharLen * _allCharLen * _allCharLen ? (int)((blockIdx.x / (_allCharLen * _allCharLen * _allCharLen)) % _allCharLen) : 0;
	digit[6] = blockIdx.x >= _allCharLen * _allCharLen ? (int)((blockIdx.x / (_allCharLen * _allCharLen)) % _allCharLen) : 0;
	digit[5] = blockIdx.x >= _allCharLen ? (int)((blockIdx.x / _allCharLen) % _allCharLen) : 0;
	digit[4] = (int)(blockIdx.x % _allCharLen);
	digit[3] = threadIdx.x;
	digit[2] = 0;
	digit[1] = 0;
    digit[0] = 0;

	while (digit[2] < _allCharLen) {
        while(digit[1] < _allCharLen) {
		    for (int i = 0; digit[1] < _allCharLen; digit[0]++, ++i) {
			    candidate[0] = s_alphabet[digit[0]];

                for(int k = 4; k <= 8; k++) {
			        for (int j = 1; j < k; j++) {
				        candidate[j] = s_alphabet[digit[j]];
                    }
                    candidate[k] = '\0';

                    if (!my_comp(_password, candidate, 8)) {
				        my_strcpy(_found, candidate);
				        _found[k] = '\0';
				        return;
			        }
			    }
		    }
            ++digit[1];
            digit[0] = 0;
        }
		++digit[2];
        digit[1] = 0;
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
    CUDA_CHECK(cudaMalloc((void**)&device_found, sizeof(char) * 8 + 1));
	CUDA_CHECK(cudaMemcpy(device_allCharacters, passwordBF.allChar, sizeof(char) * host_AllCharLen + 1, cudaMemcpyHostToDevice));

    for(auto password : passwordBF.passwordsHash) {

        std::cout << ++progressCount << "th password(" << password << ") on progress..." << std::endl;

        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMalloc((void**)&device_Password, sizeof(char) * password.length() + 1));
        CUDA_CHECK(cudaMemcpy(device_Password, password.c_str(), sizeof(char) * password.length() + 1, cudaMemcpyHostToDevice));

        dim3 threadsPerBlock(host_AllCharLen, 1);
	    dim3 blocksPerGrid((int)std::pow((float)host_AllCharLen, (float)(3)), 1);

        bruteforce<<<blocksPerGrid, threadsPerBlock, sizeof(char) * host_AllCharLen >>>(device_Password, device_allCharacters, device_found, host_AllCharLen, 0);
        CUDA_CHECK(cudaMemcpy(host_found, device_found, sizeof(char) * 8 + 1, cudaMemcpyDeviceToHost));

        result = host_found;

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
