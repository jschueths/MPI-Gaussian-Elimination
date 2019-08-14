
find_package(MPI REQUIRED)

add_executable(gaussian-elimination
    main.cpp
    gaussian.cpp gaussian.hpp
    codetimer.cpp codetimer.hpp
)
target_link_libraries(gaussian-elimination MPI::MPI_CXX)

