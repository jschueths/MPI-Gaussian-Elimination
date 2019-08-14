#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <vector>

#include <mpi.h>

// Sorts the input row into chunks to be scattered two all the processors.
void sortByProcess(std::vector<double> list1, std::vector<double>& list2, size_t size);

// Swaps two rows.
void swap(std::vector<std::vector<double>>& list, size_t count, size_t row1, size_t row2);

class GaussianEliminator {
    public:
        using DataType = std::vector<std::vector<double>>;
        GaussianEliminator(DataType data, size_t rows, size_t cols, int id, int s)
            : m_data(std::move(data)), m_rows(rows), m_cols(cols), m_id(id), size(s), m_send_buffer(rows)
        {}
        double determinant() const {
            return m_determinant;
        }
        void operator()() {
            int cur_control = 0;
            for(size_t i = 0; i < m_rows; i++) {
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
                // Find the row to swap with.
                size_t rowSwap;
                if(cur_control == m_id) {
                    rowSwap = cur_row;
                    double max = m_data[cur_index][cur_row];
                    // Find the row to swap with.
                    for(size_t j = cur_row + 1; j < m_rows; j++) {
                        if(m_data[cur_index][j] > max) {
                            rowSwap = j;
                            max = m_data[cur_index][j];
                        }
                    }
                }

                // Find out if you need to swap and then act accordingly.
                MPI_Bcast(&rowSwap, sizeof(size_t), MPI_BYTE, cur_control, MPI_COMM_WORLD);
                if(rowSwap != cur_row) {
                    swap(m_data, m_cols, cur_row, rowSwap);
                    swaps++;
                }

                if(cur_control == m_id) {
                    // Generate the coefficients.
                    for(size_t j = cur_row; j < m_rows; j++) {
                        m_send_buffer[j] = m_data[cur_index][j] / m_data[cur_index][cur_row];
                    }
                }
                // Send and recv the coefficients.
                MPI_Bcast(m_send_buffer.data(), m_rows, MPI_DOUBLE, cur_control, MPI_COMM_WORLD);
                // Apply the coefficients to the data.
                for(size_t j = 0; j < m_data.size(); j++) {
                    for(size_t k = cur_row + 1; k < m_rows; k++) {
                        m_data[j][k] -= m_data[j][cur_row] * m_send_buffer[k];
                    }
                }

                // Update the determinant value.
                if(cur_control == m_id) {
                    det_val = det_val * m_data[cur_index][cur_row];
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
            MPI_Reduce(&det_val, &m_determinant, 1, MPI_DOUBLE, MPI_PROD, 0, MPI_COMM_WORLD);
            // If we did an odd number of row swaps, negate the determinant.
            if(swaps % 2) {
                m_determinant = -m_determinant;
            }
        }

    private:
        std::vector<std::vector<double>> m_data;
        size_t m_rows;
        size_t m_cols;
        int m_id;
        int size;
        std::vector<double> m_send_buffer;
        size_t cur_row{0};
        size_t swaps{0};
        double det_val{1};
        size_t cur_index{0};
        double m_determinant;

        void log() {
            std::cout << std::endl;
        }

        template<class T, class... Args>
        void log(T t, Args... args) {
            if(m_id == 0) {
                std::cout << t;
                log(args...);
            }
        }

};
#endif // GAUSSIAN_HPP
