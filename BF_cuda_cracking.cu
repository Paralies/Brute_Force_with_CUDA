#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "BF_passwords.hpp"

#define MAX_PASSWORD_LEN 8
//===========================================================================================
// Original code from github: https://github.com/hyunsooda/Parallel-Brute-Force-Algorithm.git
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
//===========================================================================================

__device__ int comp_gen(char* _str1, char* _str2) { // Compare two strings
	int flag = 0;

	for (int i = 0; ((_str1[i] != '\0') || (_str2[i] != '\0')); i++) { // Check if it meets end of the string
		if ((_str1[i] != _str2[i])) {
			flag = 1;
			break;
		}
	}

	return flag;
}

// Brute force functions for each length of passwords
__global__ void bruteforce_4_and_5(char* _password, char* _allChar, char* _found, int _allCharLen, long long int next);
__global__ void bruteforce_6(char* _password, char* _allChar, char* _found, int _allCharLen, long long int next);
__global__ void bruteforce_7(char* _password, char* _allChar, char* _found, int _allCharLen, long long int next);
__global__ void bruteforce_8(char* _password, char* _allChar, char* _found, int _allCharLen, long long int next);

// Function to initiate cracking password by brute force
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

	// Open log file and make passwords
    logFile.open("bruteForce.log");
    passwordBF.make_password();

    CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	// Assign characters and password_found variable which will be used in GPU device
    CUDA_CHECK(cudaMalloc((void**)&device_allCharacters, sizeof(char) * host_AllCharLen + 1));
    CUDA_CHECK(cudaMalloc((void**)&device_found, sizeof(char) * MAX_PASSWORD_LEN + 1));
	CUDA_CHECK(cudaMemcpy(device_allCharacters, passwordBF.allChar, sizeof(char) * host_AllCharLen + 1, cudaMemcpyHostToDevice));

    for(auto password : passwordBF.passwordsHash) {

        std::cout << ++progressCount << "th password(" << password << ") on progress..." << std::endl;

		// Assign password variable which will be used in GPU device and start to check process time
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMalloc((void**)&device_Password, sizeof(char) * password.length() + 1));
        CUDA_CHECK(cudaMemcpy(device_Password, password.c_str(), sizeof(char) * password.length() + 1, cudaMemcpyHostToDevice));

        dim3 threadsPerBlock(host_AllCharLen, 1);
	    dim3 blocksPerGrid((int)std::pow((float)host_AllCharLen, (float)(4)), 1);

		// Brute force 4 and 5 letters
        bruteforce_4_and_5<<<blocksPerGrid, threadsPerBlock, sizeof(char) * host_AllCharLen >>>(device_Password, device_allCharacters, device_found, host_AllCharLen, 0);
        CUDA_CHECK(cudaMemcpy(host_found, device_found, sizeof(char) * MAX_PASSWORD_LEN + 1, cudaMemcpyDeviceToHost));

		result = host_found;
		
		// Brute force 6 letters
		if(result.compare(password) != 0) {
			bruteforce_6<<<blocksPerGrid, threadsPerBlock, sizeof(char) * host_AllCharLen >>>(device_Password, device_allCharacters, device_found, host_AllCharLen, 0);
			CUDA_CHECK(cudaMemcpy(host_found, device_found, sizeof(char) * MAX_PASSWORD_LEN + 1, cudaMemcpyDeviceToHost));

			result = host_found;
		}
		
		// Brute force 7 letters
		if(result.compare(password) != 0) {
			bruteforce_7<<<blocksPerGrid, threadsPerBlock, sizeof(char) * host_AllCharLen >>>(device_Password, device_allCharacters, device_found, host_AllCharLen, 0);
			CUDA_CHECK(cudaMemcpy(host_found, device_found, sizeof(char) * MAX_PASSWORD_LEN + 1, cudaMemcpyDeviceToHost));

			result = host_found;
		}
				
		
		// Brute force 8 letters
		if(result.compare(password) != 0) {
			bruteforce_8<<<blocksPerGrid, threadsPerBlock, sizeof(char) * host_AllCharLen >>>(device_Password, device_allCharacters, device_found, host_AllCharLen, 0);
			CUDA_CHECK(cudaMemcpy(host_found, device_found, sizeof(char) * MAX_PASSWORD_LEN + 1, cudaMemcpyDeviceToHost));

			result = host_found;
		}

		std::cout << "result: " << result << std::endl;
        if(result.compare(password) == 0) {
            CUDA_CHECK(cudaEventRecord(stop)); // Record stop time of progress
		    cudaEventSynchronize(stop);

		    CUDA_CHECK(cudaEventElapsedTime(&workTime, start, stop));
		    logFile << progressCount << "th password: " << result << "(time consumed=" << workTime << "ms" << ")" << std::endl; // Save found password and time consumed in the log file
        }

        CUDA_CHECK(cudaFree(device_Password));
    }

    CUDA_CHECK(cudaFree(device_allCharacters));
    logFile.close();
}

__global__ void bruteforce_4_and_5(char* _password, char* _allChar, char* _found, int _allCharLen, long long int next) {
	
	extern __shared__ char s_alphabet[];

	char candidate[MAX_PASSWORD_LEN + 1];
	int digit[8] = { 0, };

	// Assign characters in shared memory variable
	for (int i = 0; i < _allCharLen; i++)
		s_alphabet[i] = _allChar[i];
	
	// Set the threads to find the password in parallel process
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

	char candidate[MAX_PASSWORD_LEN + 1];
	int digit[8] = { 0, };

	// Assign characters in shared memory variable
	for (int i = 0; i < _allCharLen; i++)
		s_alphabet[i] = _allChar[i];

	// Set the threads to find the password in parallel process
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

__global__ void bruteforce_7(char* _password, char* _allChar, char* _found, int _allCharLen, long long int next) {
	
	extern __shared__ char s_alphabet[];

	char candidate[MAX_PASSWORD_LEN + 1];
	int digit[8] = { 0, };

	// Assign characters in shared memory variable
	for (int i = 0; i < _allCharLen; i++)
		s_alphabet[i] = _allChar[i];

	// Set the threads to find the password in parallel process
	digit[6] = 0;
	digit[5] = 0;
	digit[4] = blockIdx.x >= _allCharLen * _allCharLen * _allCharLen ? (int)((blockIdx.x / (_allCharLen * _allCharLen * _allCharLen)) % _allCharLen) : 0;
	digit[3] = blockIdx.x >= _allCharLen * _allCharLen ? (int)((blockIdx.x / (_allCharLen * _allCharLen)) % _allCharLen) : 0;
	digit[2] = blockIdx.x >= _allCharLen ? (int)((blockIdx.x / _allCharLen) % _allCharLen) : 0;
	digit[1] = (int)(blockIdx.x % _allCharLen);
	digit[0] = threadIdx.x;

	// 7 characters
	for(; digit[6] < _allCharLen; digit[6]++) {
		for(; digit[5] < _allCharLen; digit[5]++) {
			for (int j = 0; j < 5; j++) {
				candidate[j] = s_alphabet[digit[j]];
			}
			candidate[5] = s_alphabet[digit[5]];
			candidate[6] = s_alphabet[digit[6]];
        	candidate[7] = '\0';

    		if (!comp_gen(_password, candidate)) {
				my_strcpy(_found, candidate);
				_found[7] = '\0';
				return;
			}
		}
	}
}

__global__ void bruteforce_8(char* _password, char* _allChar, char* _found, int _allCharLen, long long int next) {
	
	extern __shared__ char s_alphabet[];

	char candidate[MAX_PASSWORD_LEN + 1];
	int digit[8] = { 0, };

	// Assign characters in shared memory variable
	for (int i = 0; i < _allCharLen; i++)
		s_alphabet[i] = _allChar[i];

	// Set the threads to find the password in parallel process
	digit[7] = 0;
	digit[6] = 0;
	digit[5] = 0;
	digit[4] = blockIdx.x >= _allCharLen * _allCharLen * _allCharLen ? (int)((blockIdx.x / (_allCharLen * _allCharLen * _allCharLen)) % _allCharLen) : 0;
	digit[3] = blockIdx.x >= _allCharLen * _allCharLen ? (int)((blockIdx.x / (_allCharLen * _allCharLen)) % _allCharLen) : 0;
	digit[2] = blockIdx.x >= _allCharLen ? (int)((blockIdx.x / _allCharLen) % _allCharLen) : 0;
	digit[1] = (int)(blockIdx.x % _allCharLen);
	digit[0] = threadIdx.x;

	// 8 characters
	for(; digit[7] < _allCharLen; digit[7]++) {
		for(; digit[6] < _allCharLen; digit[6]++) {
			for(; digit[5] < _allCharLen; digit[5]++) {
				for (int j = 0; j < 5; j++) {
					candidate[j] = s_alphabet[digit[j]];
				}
				candidate[5] = s_alphabet[digit[5]];
				candidate[6] = s_alphabet[digit[6]];
				candidate[7] = s_alphabet[digit[7]];
        		candidate[8] = '\0';

    			if (!comp_gen(_password, candidate)) {
					my_strcpy(_found, candidate);
					_found[8] = '\0';
					return;
				}
			}
		}
	}
}