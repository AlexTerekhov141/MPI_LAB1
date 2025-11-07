#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double now_seconds3() { return MPI_Wtime(); }

void init_matrix3(double *A, int N, unsigned int seed) {
    srand(seed);
    for (long i = 0; i < (long)N * N; ++i) A[i] = ((double)rand() / RAND_MAX);
}

void init_vector3(double *x, int N, unsigned int seed) {
    srand(seed + 123);
    for (int i = 0; i < N; ++i) x[i] = ((double)rand() / RAND_MAX);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int N = atoi(argv[1]);
    int runs = 100;
    int q = (int)round(sqrt((double)size));
    double *A = NULL;
    double *x = NULL;
    if (rank == 0) {
        A = malloc(sizeof(double) * (long)N * N);
        x = malloc(sizeof(double) * N);
        if (!A || !x) {
            perror("root malloc");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        init_matrix3(A, N, 42);
        init_vector3(x, N, 314);
    }
    MPI_Comm cart;
    int dims[2] = {q, q};
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart);
    int coords[2];
    MPI_Cart_coords(cart, rank, 2, coords);
    int prow = coords[0], pcol = coords[1];
    int bs = N / q;
    double *Ablock = malloc(sizeof(double) * (long)bs * bs);
    double *xlocal = malloc(sizeof(double) * bs);
    double *ylocal = malloc(sizeof(double) * bs);
    if (!Ablock || !xlocal || !ylocal) {
        perror("malloc block");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            int crd[2];
            MPI_Cart_coords(cart, r, 2, crd);
            int rs = crd[0] * bs;
            int cs = crd[1] * bs;
            double *tmp = malloc(sizeof(double) * bs * bs);
            for (int i = 0; i < bs; ++i) {
                for (int j = 0; j < bs; ++j) {
                    tmp[i * bs + j] = A[(rs + i) * N + (cs + j)];
                }
            }
            if (r == 0) {
                memcpy(Ablock, tmp, sizeof(double) * bs * bs);
            } else {
                MPI_Send(tmp, bs * bs, MPI_DOUBLE, r, 11, MPI_COMM_WORLD);
            }
            free(tmp);
        }
    } else {
        MPI_Recv(Ablock, bs * bs, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            int crd[2];
            MPI_Cart_coords(cart, r, 2, crd);
            int cs = crd[1] * bs;
            double *tmp = malloc(bs * sizeof(double));
            memcpy(tmp, x + cs, bs * sizeof(double));
            if (r == 0) {
                memcpy(xlocal, tmp, bs * sizeof(double));
            } else {
                MPI_Send(tmp, bs, MPI_DOUBLE, r, 22, MPI_COMM_WORLD);
            }
            free(tmp);
        }
    } else {
        MPI_Recv(xlocal, bs, MPI_DOUBLE, 0, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    double best = 1e300;
    for (int run = 0; run < runs; ++run) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = now_seconds3();

        for (int i = 0; i < bs; ++i) {
            double s = 0.0;
            for (int j = 0; j < bs; ++j) s += Ablock[i * bs + j] * xlocal[j];
            ylocal[i] = s;
        }
        
        int color = prow;
        MPI_Comm row_comm;
        MPI_Comm_split(cart, color, pcol, &row_comm);
        double *yrow = NULL;
        if (pcol == 0) yrow = malloc(sizeof(double) * bs);
        MPI_Reduce(ylocal, yrow, bs, MPI_DOUBLE, MPI_SUM, 0, row_comm);
        
        if (pcol == 0) {
            if (rank == 0) {
                
                double *yfull = malloc(sizeof(double) * N);
                
                for (int i = 0; i < bs; ++i) yfull[i] = yrow[i];
                for (int pr = 1; pr < q; ++pr) {
                    int src_coords[2] = {pr, 0};
                    int src;
                    MPI_Cart_rank(cart, src_coords, &src);
                    MPI_Recv(yfull + pr * bs, bs, MPI_DOUBLE, src, 33, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                free(yfull);
            } else {
                
                int dest_coords[2] = {0, 0};
                int dest;
                MPI_Cart_rank(cart, dest_coords, &dest);
                MPI_Send(yrow, bs, MPI_DOUBLE, dest, 33, MPI_COMM_WORLD);
            }
        }
        if (pcol == 0) free(yrow);
        MPI_Comm_free(&row_comm);
        double t1 = now_seconds3();
        double elapsed_local = t1 - t0, elapsed_max;
        MPI_Reduce(&elapsed_local, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("MPI_block,%d,%d,%d,%.9f\n", N, run, size, elapsed_max);
            if (elapsed_max < best) best = elapsed_max;
        }
    }
    if (rank == 0) fprintf(stderr, "MPI_block summary: N=%d P=%d runs=%d best=%.9f\n", N, size, runs, best);
    free(Ablock);
    free(xlocal);
    free(ylocal);
    if (rank == 0) {
        free(A);
        free(x);
    }
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
