/*************************************************
 * Este archivo fue escrito como ejemplo en el curso Applied HPC,
 * de la Universidad de Ingeniería y Tecología (UTEC)
 * Material es de libre uso, entendiendo que debe contener este encabezado
 * UTEC no se responsabiliza del uso particular del código
 *
 * Autor:       Jose Fiestas (UTEC)
 * contacto:    jfiestas@utec.edu.pe
 * objetivo:    Multiplicacion de matrices con OpenMP
 *		compilar asi:
 *		g++ -O3 -fopenmp -march=native -o mm_openmp mm_openmp.cpp
 * contenido:   código fuente en C+/OMP
  *************************************************/

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <iostream>
#include <list>
#include <numeric>
#include <fstream>

int main(int argc, char** argv) {
    //int N = 4096;
    std::list<int> n = {1024, 2048, 4096, 8192, 16384};
    std::list<int> threads = {0, 2, 4, 8, 16,24};
    //int threads = 0; // opcional
    //if (argc >= 2) N = std::atoi(argv[1]);
    //if (argc >= 3) threads = std::atoi(argv[2]);
    //if (threads > 0) omp_set_num_threads(threads);

    // Cambio de tamaños de matrices
    std::list<double> time;
    std::list<double> speedup;
    std::ofstream time_r;
    time_r.open("time_r.csv");
    std::ofstream speedup_r;
    speedup_r.open("speedup_r.csv");

    time_r << "N,0,2,4,8,16,24\n";
    speedup_r << "N,2,4,8,24\n";
    
    for (int N : n) {
        std::vector<double> A((size_t)N*N), B((size_t)N*N), C((size_t)N*N, 0.0);
        std::mt19937_64 g(42);
        std::uniform_real_distribution<double> d(-1.0,1.0);
        time_r << N << ",";
        speedup_r << N << ",";
        for (size_t i=0;i<(size_t)N*N;++i){ A[i]=d(g); B[i]=d(g); }
        for (int t : threads) {
            omp_set_num_threads(t);
            // list for mean calculation
            speedup.clear();
            time.clear();
            for (int b = 0; b < 10; b++){
                double t0 = omp_get_wtime();
                // Paralelizamos por filas de C (cada hilo tiene una fila exclusiva)
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < N; ++i) {
                    const double* Ai = &A[(size_t)i * N];
                    double* Ci       = &C[(size_t)i * N];
                    for (int k = 0; k < N; ++k) {
                        const double aik = Ai[k];
                        const double* Bk = &B[(size_t)k * N];   // fila k-ésima de B (contigua)
                        // Recorrido contiguo de B y C: se vectoriza bien
                        #pragma omp simd
                        for (int j = 0; j < N; ++j) {
                            Ci[j] += aik * Bk[j];
                        }
                    }
                }
                double t1 = omp_get_wtime();
                const double secs = t1 - t0;
                const double gflops = (2.0 * (double)N * N * N) / (secs * 1e9);
                // add result to the list
                time.emplace_back(secs);
                speedup.push_back(gflops);
            }
            // Time mean calculation
            float sum_t = std::accumulate(time.begin(), time.end(), 0.0);
            int n_t = time.size();
            float mean_t = 0.0;
            mean_t = sum_t / n_t;
            // Speedup mean calculation
            float sum_s = std::accumulate(speedup.begin(), speedup.end(), 0.0);
            int n_s = speedup.size();
            float mean_s = 0.0;
            mean_s = sum_s / n_s;
            time_r << mean_t << ",";
            speedup_r << mean_s << ",";
            //std::printf("\nN=%d, threads=%d\n", N, t);
            //std::printf("Tiempo GEMM: %.6f s\n", mean_t);
            //std::printf("Rendimiento: %.2f GFLOP/s\n", mean_s);
        }  
        time_r << "\n";
        speedup_r << "\n";
    }
    time_r.close();
    speedup_r.close();
    return 0;
}

