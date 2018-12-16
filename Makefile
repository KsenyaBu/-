all: clean main.c
	mpicc main.c -O3 -DNDEBUG -g -std=c99 -openmp -o main
#	mpicc main.c -O3 -DNDEBUG -g -std=c99 -fopenmp -o main
clean:
	rm -rf main.dSYM*
	rm -rf main