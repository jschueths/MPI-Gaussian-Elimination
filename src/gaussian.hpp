#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <vector>

// Sorts the input row into chunks to be scattered two all the processors.
void sortByProcess(std::vector<double> list1, double* list2, size_t count, size_t size);

// Swaps two rows.
void swap(std::vector<std::vector<double>>& list, size_t count, size_t row1, size_t row2);

#endif // GAUSSIAN_HPP
