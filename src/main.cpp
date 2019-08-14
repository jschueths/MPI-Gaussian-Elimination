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
    int cur_control = 0;

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
    MPI_Barrier(MPI_COMM_WORLD);
    auto sTime = MPI_Wtime();
    timer.start();

    // Actual Gaussian code here.
    /*Algorithm for Gaussian elimination (with pivoting):
    Start with all the numbers stored in our NxN matrix A.
    For each column p, we do the following (p=1..N)
        First make sure that a(p,p) is non-zero and preferably large:
        Look at the rows in our matrix below row p.  Look at the p'th
        term in each row.  Select the row that has the largest absolute
        value in the p'th term, and swap the p'th row with that one.
        (optionally, you can only bother to do the above step if
        a(p,p) is zero).
    If we were fortunate enough to get a non-zero value for a(p,p),
    then proceed with the following for loop:
    For each row r below p, we do the following (r=p+1..N)
        row(r) = row(r)  -  (a(r,p) / a(p,p)) * row(p)
    End For
    */
    std::vector<double> send_buffer(num_rows);
    size_t cur_row = 0;
    size_t swaps = 0;
    double det_val = 1;
    size_t cur_index = 0;
    for(size_t i = 0; i < num_rows; i++) {
        // Find the row to swap with.
        size_t rowSwap;
        if(cur_control == rank) {
            rowSwap = cur_row;
            double max = data[cur_index][cur_row];
            // Find the row to swap with.
            for(size_t j = cur_row + 1; j < num_rows; j++) {
                if(data[cur_index][j] > max) {
                    rowSwap = j;
                    max = data[cur_index][j];
                }
            }
        }

        // Find out if you need to swap and then act accordingly.
        MPI_Bcast(&rowSwap, sizeof(size_t), MPI_BYTE, cur_control, MPI_COMM_WORLD);
        if(rowSwap != cur_row) {
          swap(data, num_cols, cur_row, rowSwap);
          swaps++;
        }

        if(cur_control == rank) {
            // Generate the coefficients.
            for(size_t j = cur_row; j < num_rows; j++) {
                send_buffer[j] = data[cur_index][j] / data[cur_index][cur_row];
            }
        }
        // Send and recv the coefficients.
        MPI_Bcast(send_buffer.data(), num_rows, MPI_DOUBLE, cur_control, MPI_COMM_WORLD);
        // Apply the coefficients to the data.
        for(size_t j = 0; j < num_cols; j++) {
            for(size_t k = cur_row + 1; k < num_rows; k++) {
                data[j][k] -= data[j][cur_row] * send_buffer[k];
            }
        }

        // Update the determinant value.
        if(cur_control == rank) {
            det_val = det_val * data[cur_index][cur_row];
            cur_index++;
        }

        // Increment the row that we are looking at
        // and increment the counter that tells each process where
        // to recv from. The counter resets to zero to give us a
        // "round robin" communication pattern. Probably not very efficient,
        // but it will do for now.
        cur_control++;
        if(cur_control == size) {
            cur_control = 0;
        }
        cur_row++;
    }

    // Reduce all the determinant values from each process
    // with a multiplication operation.
    // Personally I really like the method I used to find the determinant:
    //   1. Each process just keeps multiplying the pivot value into the product.
    //   2. The reduce does a multiply on all of the individual products.
    // So there really is no extra work to find the determinant.
    double determinant;
    MPI_Reduce(&det_val, &determinant, 1, MPI_DOUBLE, MPI_PROD, 0, MPI_COMM_WORLD);
    // If we did an odd number of row swaps, negate the determinant.
    if(swaps % 2) {
        determinant = -determinant;
    }

    // End timing.
    MPI_Barrier(MPI_COMM_WORLD);
    auto eTime = MPI_Wtime();
    timer.stop();

    auto rTime = eTime - sTime;

    // If root node, output the runtime.
    if(!rank) {
        std::cout << "MPI Wall Time: " << rTime << std::endl;
        std::cout << "Root node time: " << timer.duration().count() << std::endl;
        std::cout << "Determinant value: " << determinant << std::endl;
    }

    // Finalize and exit.
    MPI_Finalize();
    return 0;
}
