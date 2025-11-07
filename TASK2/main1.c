#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static double now_seconds() { return MPI_Wtime(); }


void init_matrix(double *A, int N, unsigned int seed) {
	srand(seed);
	for (long i = 0; i < (long)N * N; ++i) { 
            A[i] = ((double)rand() / RAND_MAX);
        }
}

void init_vector(double *x, int N, unsigned int seed) {
	srand(seed+123);
	for (int i = 0; i < N; ++i) {
	    x[i] = ((double)rand() / RAND_MAX);
        }
}


int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);

	int N = atoi(argv[1]);
	int runs = 1000;


	double *A = NULL; double *x = NULL;
	if (rank==0) {
	   A = malloc(sizeof(double)*(long)N*N);
	   x = malloc(sizeof(double)*N);
	   if (!A || !x) {
	       perror("root malloc");
	       MPI_Abort(MPI_COMM_WORLD,1);
	   }
	   init_matrix(A, N, 42);
	   init_vector(x, N, 314);
	}



	int base = N / size; int rem = N % size;
	int my_rows = base + (rank < rem ? 1 : 0);
	int *sendcounts = NULL; int *displs = NULL;
	if (rank==0) {
	   sendcounts = calloc(size, sizeof(int)); displs = calloc(size, sizeof(int));
	   int off = 0;
	for (int p=0;p<size;++p) {
	     int rows_p = base + (p < rem ? 1 : 0);
	     sendcounts[p] = rows_p * N;
	     displs[p] = off;
	     off += sendcounts[p];
	}
	}


	double *localA = malloc(sizeof(double) * (long)my_rows * N);
	double *localx = malloc(sizeof(double) * N);
	double *localy = malloc(sizeof(double) * my_rows);
	if (!localA || !localx || !localy) { 
	    perror("malloc local"); 
	    MPI_Abort(MPI_COMM_WORLD,1); 
	}


	double best = 1e300;
	for (int run=0; run<runs; ++run) {
	    MPI_Barrier(MPI_COMM_WORLD);
	    double t0 = now_seconds();


	    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, localA, my_rows*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    MPI_Bcast(localx, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);



	    for (int i=0;i<my_rows;++i) {
		 double s = 0.0;
		 double *row = localA + (long)i * N;
	    for (int j=0;j<N;++j) s += row[j] * localx[j];
            	 localy[i] = s;
	    }

	    MPI_Gatherv(localy, my_rows, MPI_DOUBLE, A, sendcounts ? sendcounts : NULL, displs ? displs : NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);


	    double t1 = now_seconds();
	    double elapsed_local = t1 - t0, elapsed_max;
            MPI_Reduce(&elapsed_local, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	    if (rank==0) {
	        printf("MPI_row,%d,%d,%d,%.9f\n", N, run, size, elapsed_max);
	    if (elapsed_max < best) best = elapsed_max;
	    }
	}


	if (rank==0) fprintf(stderr, "MPI_row summary: N=%d P=%d runs=%d best=%.9f\n", N, size, runs, best);


	free(localA); free(localx); free(localy);
	if (rank==0) { free(A); free(x); free(sendcounts); free(displs); }
	MPI_Finalize();
	return 0;
}
