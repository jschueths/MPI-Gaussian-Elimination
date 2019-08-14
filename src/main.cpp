#include <fstream>
#include <vector>

#include <mpi.h>

#include "codetimer.hpp"
#include "gaussian.hpp"

auto mpi_init(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return std::make_tuple(rank, size);
}

auto scatter_data(size_t rows, size_t cols, int rank, int size, std::istream& in) {
    std::vector<std::vector<double>> data(cols, std::vector<double>(rows, 0.0));
    std::vector<double> send_buffer(rows);
    std::vector<double> recv_buffer(cols);
    std::vector<double> file_buffer(rows);
    // Scatter the data.
    for(size_t i = 0; i < rows; i++) {
        if(!rank) {
            for(auto& val : file_buffer) {
                in >> val;
            }
            sortByProcess(file_buffer, send_buffer, size);
        }
        // Scatters the data so that each process gets the next value for their columns.
        MPI_Scatter(send_buffer.data(), cols, MPI_DOUBLE, recv_buffer.data(), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for(size_t j = 0; j < cols; j++) {
            data[j][i] = recv_buffer[j];
        }
    }
    return data;
}

auto& newl(std::ostream& out) {
    out << '\n';
    return out;
}

int main(int argc, char* argv[]) {
    std::ifstream inFile;

    // Just get the initialization of the program going.
    auto [rank, size] = mpi_init(argc, argv);

    // If the input file is not given, print message and exit.
    if(argc < 2) {
        if(rank == 0) {
            std::cout << "No input file given." << newl;
        }
        MPI_Finalize();
        return 0;
    }
    // If the root node (0), then open the input file and read in the
    // number of rows.
    size_t num_rows;
    if(!rank) {
        inFile.open(argv[1]);
        inFile >> num_rows;
    }

    // Broadcasts the number of rows to each processor.
    MPI_Bcast(&num_rows, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    size_t num_cols = num_rows / static_cast<size_t>(size);
    if(rank == 0) {
        std::cout << "Scattering data..." << newl;
    };

    // Scatter the data to all processors
    CodeTimer timer;
    timer.start();
    auto data = scatter_data(num_rows, num_cols, rank, size, inFile);
    timer.stop();
    if(rank == 0) {
        std::cout << "Scattering completed in: " << timer.duration().count() << " sec" << newl;
        std::cout << "Running Gaussian Elimination..." << newl;
    }
    // Begin timing and gaussian elimination.
    GaussianEliminator gaussian(std::move(data), num_rows, num_cols, rank, size);
    MPI_Barrier(MPI_COMM_WORLD);
    auto sTime = MPI_Wtime();
    timer.start();
    gaussian();

    // End timing.
    MPI_Barrier(MPI_COMM_WORLD);
    auto eTime = MPI_Wtime();
    timer.stop();

    auto rTime = eTime - sTime;

    // If root node, output the runtime.
    if(!rank) {
        std::cout << "MPI Wall Time: " << rTime << " sec" << newl;
        std::cout << "Root node time: " << timer.duration().count() << " sec" << newl;
        std::cout << "Determinant value: " << gaussian.determinant() << "sec" << newl;
    }

    // Finalize and exit.
    MPI_Finalize();
    return 0;
}
