#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static double now_seconds2() { return MPI_Wtime(); }


void init_matrix2(double *A, int N, unsigned int seed) {
	srand(seed);
	for (long i = 0; i < (long)N * N; ++i){ 
	     A[i] = ((double)rand() / RAND_MAX);
	}
}
void init_vector2(double *x, int N, unsigned int seed) {
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
	   if (!A || !x) { perror("root malloc"); MPI_Abort(MPI_COMM_WORLD,1); }
	   init_matrix2(A, N, 42);
	   init_vector2(x, N, 314);
	}



	int base = N / size; int rem = N % size;
	int my_cols = base + (rank < rem ? 1 : 0);



	double *localAcols = malloc(sizeof(double) * (long)N * my_cols);
	double *localx = malloc(sizeof(double) * my_cols);
	double *partial_y = malloc(sizeof(double) * N);
	if (!localAcols || !localx || !partial_y) { perror("malloc local"); MPI_Abort(MPI_COMM_WORLD,1); }


	int *cols_counts = NULL; int *cols_displs = NULL;
	int *x_counts = NULL; int *x_displs = NULL;
	if (rank==0) {
	   cols_counts = calloc(size, sizeof(int)); cols_displs = calloc(size, sizeof(int));
	   x_counts = calloc(size, sizeof(int)); x_displs = calloc(size, sizeof(int));
	   int off = 0, xoff = 0;
	   for (int p=0;p<size;++p) {
	        int cp = base + (p < rem ? 1 : 0);
	        cols_counts[p] = N * cp;
		cols_displs[p] = off; off += cols_counts[p];
		x_counts[p] = cp; x_displs[p] = xoff; xoff += cp;
	   }
}


	double best = 1e300;
	for (int run=0; run<runs; ++run) {
		MPI_Barrier(MPI_COMM_WORLD);
		double t0 = now_seconds2();


		if (rank==0) {
		   double *A_cols = malloc(sizeof(double)*(long)N*N);
		   int pos = 0;
		   for (int col=0; col<N; ++col) {
		   for (int row=0; row<N; ++row) A_cols[pos++] = A[row*N + col];
		}
		   MPI_Scatterv(A_cols, cols_counts, cols_displs, MPI_DOUBLE, localAcols, N*my_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		   MPI_Scatterv(x, x_counts, x_displs, MPI_DOUBLE, localx, my_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		   free(A_cols);
		} else {
			MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, localAcols, N*my_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, localx, my_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}
		for (int i=0;i<N;++i) partial_y[i]=0.0;
		for (int c=0;c<my_cols;++c) {
		     double xc = localx[c];
		     double *colptr = localAcols + (long)c * N;
		for (int i=0;i<N;++i) partial_y[i] += colptr[i] * xc;
		}



		double *yroot = NULL;
		if (rank==0) yroot = malloc(sizeof(double)*N);
		MPI_Reduce(partial_y, yroot, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


		double t1 = now_seconds2();
		double elapsed_local = t1 - t0, elapsed_max;
		MPI_Reduce(&elapsed_local, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		if (rank==0) {
		    printf("MPI_col,%d,%d,%d,%.9f\n", N, run, size, elapsed_max);
		if (elapsed_max < best) best = elapsed_max;
		    free(yroot);
		}
	}


	if (rank==0) fprintf(stderr, "MPI_col summary: N=%d P=%d runs=%d best=%.9f\n", N, size, runs, best);


	free(localAcols); free(localx); free(partial_y);
	if (rank==0) { free(A); free(x); free(cols_counts); free(cols_displs); free(x_counts); free(x_displs); }
	   MPI_Finalize();
	return 0;
}
