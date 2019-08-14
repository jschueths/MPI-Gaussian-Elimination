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

class RootLogger {
    public:
        RootLogger(int nodeId) : mId(nodeId) {}
        void operator()(std::string_view s) {
            if(mId == 0) {
                std::cout << s << newl;
            }
        }

    private:
        int mId{0};
        static std::ostream& newl(std::ostream& out) {
            out << '\n';
            return out;
        }
};

int main(int argc, char* argv[]) {
    std::ifstream inFile;

    // Just get the initialization of the program going.
    auto [rank, size] = mpi_init(argc, argv);
    RootLogger logger(rank);
    // If the input file is not given, print message and exit.
    if(argc < 2) {
        logger("No input file given.");
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
    // Allocate the memory on each processor.
    std::vector<std::vector<double>> data(num_cols, std::vector<double>(num_rows, 0.0));
    logger("Scattering data...");
    {
        std::vector<double> send_buffer(num_rows);
        std::vector<double> recv_buffer(num_cols);
        std::vector<double> file_buffer(num_rows);
        // Scatter the data.
        for(size_t i = 0; i < num_rows; i++) {
            if(!rank) {
                for(auto& val : file_buffer) {
                    inFile >> val;
                }
                sortByProcess(file_buffer, send_buffer, size);
            }
            // Scatters the data so that each process gets the next value for their columns.
            MPI_Scatter(send_buffer.data(), num_cols, MPI_DOUBLE, recv_buffer.data(), num_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            for(size_t j = 0; j < num_cols; j++) {
                data[j][i] = recv_buffer[j];
            }
        }
    }
    logger("Running Gaussian Elimination...");
    // Begin timing.
    CodeTimer timer;
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
        std::cout << "MPI Wall Time: " << rTime << std::endl;
        std::cout << "Root node time: " << timer.duration().count() << std::endl;
        std::cout << "Determinant value: " << gaussian.determinant() << std::endl;
    }

    // Finalize and exit.
    MPI_Finalize();
    return 0;
}
