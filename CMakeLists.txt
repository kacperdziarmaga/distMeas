cmake_minimum_required(VERSION 3.25)
project(DistMeasurement LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV 4 REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

if(APPLE)
    set(BLA_VENDOR "Apple")
endif()

find_package(dlib QUIET)
if(NOT dlib_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(DLIB REQUIRED IMPORTED_TARGET dlib-1) 
endif()

set(SOURCES main.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})

target_compile_options(${PROJECT_NAME} PRIVATE 
    -Wall -Wextra -Wpedantic -Wconversion
)

if(NOT CMAKE_BUILD_TYPE MATCHES "Debug")
    target_compile_options(${PROJECT_NAME} PRIVATE -O3 -march=native)
endif()

if(dlib_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE dlib::dlib)
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE PkgConfig::DLIB)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    BLAS::BLAS
    LAPACK::LAPACK
)

if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${ACCELERATE_FRAMEWORK})
    endif()
endif()