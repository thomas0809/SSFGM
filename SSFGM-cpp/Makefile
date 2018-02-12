OS := $(shell uname)
ifeq ($(OS),Darwin)
	CC = g++-6
else
	CC = g++
endif

main: src/*.cpp src/*.h
	$(CC) src/*.cpp -O2 -fopenmp -std=c++11 -o SSFGM

clean:
	rm SSFGM
