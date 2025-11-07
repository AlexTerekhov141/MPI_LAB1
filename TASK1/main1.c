#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const char *algo_name = "blocking_send_recv";
    unsigned int base_seed = (unsigned int)time(NULL);
    long long total_points = 1000000LL;
    int runs = 100;
    unsigned long long pts_per_proc = total_points / (unsigned long long)size;
    unsigned long long rem = total_points % (unsigned long long)size;
    unsigned long long local_n = pts_per_proc + (rank < (int)rem ? 1ULL : 0ULL);
    unsigned long long global_inside;
    double best_time = 1e300;
    for (int run = 0; run < runs; ++run) {
        MPI_Barrier(MPI_COMM_WORLD);
        unsigned int seed = base_seed ^ (unsigned int)(rank * 374761393u) ^ (unsigned int)(run * 668265263u);
        unsigned long long local_inside = 0;
        double t0 = MPI_Wtime();
        for (unsigned long long i = 0; i < local_n; ++i) {
            double x = (2.0 * rand_r(&seed) / (RAND_MAX + 1.0)) - 1.0;
            double y = (2.0 * rand_r(&seed) / (RAND_MAX + 1.0)) - 1.0;
            if (x*x + y*y <= 1.0) local_inside++;
        }
        // Blocking Send/Recv
        if (rank == 0) {
            unsigned long long sum = local_inside;
            for (int src = 1; src < size; ++src) {
                unsigned long long tmp = 0;
                MPI_Recv(&tmp, 1, MPI_UNSIGNED_LONG_LONG, src, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                sum += tmp;
            }
            global_inside = sum;
        } else {
            MPI_Send(&local_inside, 1, MPI_UNSIGNED_LONG_LONG, 0, 123, MPI_COMM_WORLD);
        }
        double t1 = MPI_Wtime();
        double elapsed_local = t1 - t0;
        double elapsed_max;
        MPI_Reduce(&elapsed_local, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            double pi_est = 4.0 * (double)global_inside / (double)total_points;
            printf("%s,%llu,%d,%d,%.9f,%.12f\n",
                   algo_name, total_points, run, size, elapsed_max, pi_est);
            if (elapsed_max < best_time) best_time = elapsed_max;
        }
    }
    if (rank == 0) {
        fprintf(stderr, "%s summary: total_points=%llu P=%d runs=%d best_time=%.9f\n",
                algo_name, total_points, size, runs, best_time);
    }
    MPI_Finalize();
    return 0;
}
