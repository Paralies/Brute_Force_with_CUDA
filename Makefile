CC = g++
NVCC = nvcc
OBJS = BF_cuda_cracking.o BF_passwords.o
TARGET = BruteForce.out

all: $(TARGET)

clean:
	rm -f *.o
	rm -f $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) -o $@ $(OBJS)

BF_cuda_cracking.o: BF_passwords.hpp BF_cuda_cracking.cu
	 $(NVCC) -c -o BF_cuda_cracking.o BF_cuda_cracking.cu

BF_passwords.o: BF_passwords.hpp BF_passwords.cpp
	 $(CC) -c -o BF_passwords.o BF_passwords.cpp
