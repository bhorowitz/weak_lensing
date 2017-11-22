cmake_minimum_required (VERSION 2.8.10)

project ("HMC" CXX)

# make sure that the default is a RELEASE
if (NOT CMAKE_BUILD_TYPE)
	set (CMAKE_BUILD_TYPE RELEASE)
endif (NOT CMAKE_BUILD_TYPE)

include(cmake_settings.txt)

# compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fpic")
set(CMAKE_CXX_FLAGS_DEBUG "-g -ggdb")
if(${CMAKE_CXX_COMPILER_ID} MATCHES GNU)
	set(CMAKE_CXX_FLAGS_RELEASE "-O2 -ffast-math")
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES Intel)
	set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")
endif(${CMAKE_CXX_COMPILER_ID} MATCHES GNU)

#check for openmp
find_package(OpenMP)
if(OPENMP_FOUND)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	add_definitions(-DCOSMO_OMP)
endif(OPENMP_FOUND)

include_directories(${PROJECT_SOURCE_DIR})

#check for MPI
find_package(MPI)
if(NOT MPI_FOUND)
	message(FATAL_ERROR "MPI is needed!")
endif(NOT MPI_FOUND)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
add_definitions(-DCOSMO_MPI)
if(NOT MPI_CXX_LIBRARIES)
    #message(FATAL_ERROR "MPI found but the MPI C++ libraries not found!")
endif(NOT MPI_CXX_LIBRARIES)

#cosmopp
if(NOT COSMOPP_DIR)
    message(FATAL_ERROR "COSMOPP_DIR must be specified!")
endif(NOT COSMOPP_DIR)
include_directories(${COSMOPP_DIR}/include)
find_library(COSMOPPLIB cosmopp ${COSMOPP_DIR}/lib)
if(NOT COSMOPPLIB)
	message(FATAL_ERROR "cosmopp library not found!")
endif(NOT COSMOPPLIB)

#fftw
if(NOT FFTW_DIR)
    find_library(FFTWLIB fftw3)
else(NOT FFTW_DIR)
    include_directories(${FFTW_DIR}/include)
    find_library(FFTWLIB fftw3 ${FFTW_DIR}/lib)
endif(NOT FFTW_DIR)
if(NOT FFTWLIB)
	message(FATAL_ERROR "fftw library not found!")
endif(NOT FFTWLIB)

#lapack
if(NOT LAPACK_LIB_FLAGS)
    message(FATAL_ERROR "LAPACK_LIB_FLAGS must be specified!")
endif(NOT LAPACK_LIB_FLAGS)
add_definitions(-DCOSMO_LAPACK)


enable_testing()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/hmc_results)

set(LIB_FILES power_spectrum.cpp lin_alg.cpp delta_vector.cpp)
add_library(weak_lensing STATIC ${LIB_FILES})

add_executable(fftw_test fftw_test.cpp)
target_link_libraries(fftw_test ${COSMOPPLIB})
target_link_libraries(fftw_test ${FFTWLIB})
target_link_libraries(fftw_test ${MPI_CXX_LIBRARIES})
target_link_libraries(fftw_test -dynamic)
install(TARGETS fftw_test DESTINATION bin)

add_executable(test_power_spectrum test_power_spectrum.cpp)
target_link_libraries(test_power_spectrum weak_lensing)
target_link_libraries(test_power_spectrum ${COSMOPPLIB})
target_link_libraries(test_power_spectrum ${FFTWLIB})
target_link_libraries(test_power_spectrum ${MPI_CXX_LIBRARIES})
target_link_libraries(test_power_spectrum ${LAPACK_LIB_FLAGS})
target_link_libraries(test_power_spectrum -dynamic)
install(TARGETS test_power_spectrum DESTINATION bin)

add_executable(lin_lbfgs_2d lin_lbfgs_2d.cpp)
target_link_libraries(lin_lbfgs_2d weak_lensing)
target_link_libraries(lin_lbfgs_2d ${COSMOPPLIB})
target_link_libraries(lin_lbfgs_2d ${FFTWLIB})
target_link_libraries(lin_lbfgs_2d ${MPI_CXX_LIBRARIES})
target_link_libraries(lin_lbfgs_2d ${LAPACK_LIB_FLAGS})
target_link_libraries(lin_lbfgs_2d -dynamic)
install(TARGETS lin_lbfgs_2d DESTINATION bin)
