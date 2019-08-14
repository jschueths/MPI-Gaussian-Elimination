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

#include "gaussian.hpp"

void sortByProcess(std::vector<double> list2, double* list1, size_t count, size_t size) {
    size_t index = 0;
    for(size_t i = 0; i < size; i++) {
        for(size_t j = i; j < count; j += size) {
            list1[index] = list2[j];
            index++;
        }
    }
}

void swap(std::vector<std::vector<double>>& list, size_t count, size_t row1, size_t row2) {
    if(row1 == row2) {
        return;
    }
    for(size_t i = 0; i < count; i++) {
        std::swap(list[i][row1], list[i][row2]);
    }
}
