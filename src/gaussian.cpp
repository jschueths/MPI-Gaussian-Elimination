// Joshua McCarville-Schueths
// Student ID: 12122858
// gaussian.cpp
//
// This program is an implementation of parallel gaussian elimination.
//

#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#include <mpi.h>

// Sorts the input row into chunks to be scattered two all the processors.
void sortByProcess(std::vector<double> list1, double* list2, size_t count, size_t size);

// Swaps two rows.
void swap(double** list, size_t count, size_t row1, size_t row2);

int main(int argc, char * argv[]) {
    double sTime, eTime, rTime;
    int rank;
    int size;
    std::ifstream inFile;
    size_t num_rows = 3200;
    size_t num_cols = 3200;
    int cur_control = 0;
    double determinant;
    std::vector<double> file_buffer;

    // Just get the initialization of the program going.
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // If the input file is not given, print message and exit.
    if(argc < 2) {
        std::cout << "No input file given." << std::endl;
        MPI_Finalize();
        return 0;
    }
    // If the root node (0), then open the input file and read in the
    // number of rows.
    if(!rank) {
        inFile.open(argv[1]);
        inFile >> num_rows;
        file_buffer.resize(num_rows);
    }

    auto send_buffer = new double[num_rows];
    // Broadcasts the number of rows to each processor.
    MPI_Bcast (&num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    num_cols = num_rows / static_cast<size_t>(size);
    // Allocate the memory on each processor.
    auto data = new double*[num_cols];
    for(size_t i = 0; i < num_cols; i++) {
        data[i] = new double[num_rows];
    }
    for(size_t i = 0; i < num_cols; i++) {
        for(size_t j = 0; j < num_rows; j++) {
            data[i][j] = 0;
        }
    }
    auto recv_buffer = new double[num_cols];
    // Scatter the data.
    for(size_t i = 0; i < num_rows; i++) {
        if(!rank) {
            for(size_t j = 0; j < num_rows; j++) {
                inFile >> file_buffer[j];
            }
            sortByProcess(file_buffer, send_buffer, num_rows, size);
        }
        // Scatters the data so that each process gets the next value for their columns.
        MPI_Scatter(send_buffer, num_cols, MPI_DOUBLE, recv_buffer, num_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for(size_t j = 0; j < num_cols; j++) {
            data[j][i] = recv_buffer[j];
        }
    }
    delete [] recv_buffer;
    delete [] send_buffer;
    // Begin timing.
    MPI_Barrier(MPI_COMM_WORLD);
    sTime = MPI_Wtime();

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
    send_buffer = new double[num_rows];
    int cur_row = 0;
    int swaps = 0;
    double det_val = 1;
    int cur_index = 0;
    for(size_t i = 0; i < num_rows; i++) {
        // Find the row to swap with.
        int rowSwap;
        if(cur_control == rank) {
            rowSwap = cur_row;
            double max = data[cur_index][cur_row];
            // Find the row to swap with.
            for(int j = cur_row + 1; j < num_rows; j++) {
                if(data[cur_index][j] > max) {
                    rowSwap = j;
                    max = data[cur_index][j];
                }
            }
        }

        // Find out if you need to swap and then act accordingly.
        MPI_Bcast(&rowSwap, 1, MPI_INT, cur_control, MPI_COMM_WORLD);
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
        MPI_Bcast(send_buffer, num_rows, MPI_DOUBLE, cur_control, MPI_COMM_WORLD);
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
    MPI_Reduce(&det_val, &determinant, 1, MPI_DOUBLE, MPI_PROD, 0, MPI_COMM_WORLD);
    // If we did an odd number of row swaps, negate the determinant.
    if(swaps % 2) {
        determinant = -determinant;
    }

    // End timing.
    MPI_Barrier(MPI_COMM_WORLD);
    eTime = MPI_Wtime();
    rTime = eTime - sTime;

    // If root node, output the runtime.
    if(!rank) {
        std::cout << "Run Time: " << rTime << std::endl;
        std::cout << "Determinant value: " << determinant << std::endl;
    }

    // A bit of house cleaning.
    delete [] send_buffer;
    for(size_t i = 0; i < num_cols; i++){
        delete [] data[i];
    }
    delete [] data;

    // Finalize and exit.
    MPI_Finalize();
    return 0;
}

void sortByProcess(std::vector<double> list2, double* list1, size_t count, size_t size) {
    size_t index = 0;
    for(size_t i = 0; i < size; i++) {
        for(size_t j = i; j < count; j += size) {
            list1[index] = list2[j];
            index++;
        }
    }
}

void swap(double** list, size_t count, size_t row1, size_t row2) {
    if(row1 == row2) {
        return;
    }
    for(size_t i = 0; i < count; i++) {
        std::swap(list[i][row1], list[i][row2]);
    }
}
