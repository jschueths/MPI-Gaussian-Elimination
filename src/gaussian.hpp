#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <vector>

#include <mpi.h>

// Sorts the input row into chunks to be scattered two all the processors.
void sortByProcess(std::vector<double> list1, std::vector<double>& list2, size_t size);

class GaussianEliminator {
    public:
        using DataType = std::vector<std::vector<double>>;
        GaussianEliminator(DataType data, size_t rows, size_t cols, int id, int s);
        double determinant() const;
        void operator()();

    private:
        std::vector<std::vector<double>> m_data;
        size_t m_rows{0};
        size_t m_cols{0};
        int m_id{0};
        int size{0};
        std::vector<double> m_send_buffer;
        double m_determinant{0.0};
};
#endif // GAUSSIAN_HPP
