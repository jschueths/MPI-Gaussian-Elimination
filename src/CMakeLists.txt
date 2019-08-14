
find_package(MPI REQUIRED)

add_executable(gaussian-elimination gaussian.cpp)
target_link_libraries(gaussian-elimination MPI::MPI_CXX)

