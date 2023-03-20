CC = g++
CFLAGS = -g -Wall
OBJS = main.o BF_passwords.o
TARGET = BruteForce.out

all: $(TARGET)

clean:
	rm -f *.o
	rm -f $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS)

main.o: BF_passwords.hpp BF_passwords.cpp main.cpp
BF_passwords: BF_passwords.hpp BF_passwords.cpp