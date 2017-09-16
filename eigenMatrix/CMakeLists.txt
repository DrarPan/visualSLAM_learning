cmake_minimum_required(VERSION 2.4.6)
project(slam_learning)
include(CheckCXXCompilerFlag)

set(CMAKE_BUILD_TYPE "Release")

CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++=0x" COMPILER_SUPPORTS_CXXOX)

if(COMPILER_SUPPORTS_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
     message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3 -fopenmp" )

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(Sophus REQUIRED)

find_package(Ceres REQUIRED)

set(CMAKE_PREFIX_PATH "/home/hitrobot/Programs/opencv-3.1.0/build")
#find_package(OpenCV 3.1 COMPONENTS core highgui imgproc features2d calib3d REQUIRED)
find_package(OpenCV 3.1)
set(OpenCV_INCLUDE_DIRS "/home/hitrobot/Programs/opencv-3.1.0/include")
#find_package(OpenCV REQUIRED)

find_package(DBoW3 REQUIRED)
find_package(G2O REQUIRED)
find_package(CSparse REQUIRED)


include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${CSPARSE_INCLUDE_DIR})
include_directories(${G2O_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${DBoW3_INCLUDE_DIRS})
include_directories("/usr/include/eigen3")

message("xxxxxxxxxxxxxxxxxxxxxxx")

#include_directories("/home/hitrobot/Programs/g2o/EXTERNAL/csparse/")

#add_executable(eigenMatrix eigenMatrix.cpp)
#target_link_libraries(eigenMatrix ${Sophus_LIBRARIES})

#add_executable(ceres_curve_fitting ceres_curve_fitting.cpp)
#target_link_libraries(ceres_curve_fitting ${OpenCV_LIBS} ${CERES_LIBRARIES} g2o_core g2o_stuff)

##add_executable(opencv_part opencv_part.cpp)
##target_link_libraries(opencv_part ${OpenCV_LIBS})

#add_executable(LKOpticalFlow useLK.cpp)
#target_link_libraries(LKOpticalFlow ${OpenCV_LIBS})

#add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
#target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

#add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
#target_link_libraries(pose_estimation_3d2d ${OpenCV_LIBS} ${CSPARSE_LIBRARY} g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)

#add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp slambase.hpp)
#target_link_libraries(pose_estimation_3d3d ${OpenCV_LIBS} ${CSPARSE_LIBRARY} g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)

#add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
#target_link_libraries(pose_estimation_3d2d ${OpenCV_LIBS} ${CSPARSE_LIBRARY} g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)

#add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp slambase.hpp)
#target_link_libraries(pose_estimation_3d3d ${OpenCV_LIBS} ${CSPARSE_LIBRARY} g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)

add_executable(direct_sparse direct_sparse.cpp)
target_link_libraries(direct_sparse ${OpenCV_LIBS} ${CSPARSE_LIBRARY} g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)

add_executable(dbow3training dbow3training.cpp)
target_link_libraries(dbow3training ${OpenCV_LIBS} ${DBoW3_LIBS})

#add_executable(triangulation triangulation.cpp extra.hpp)
#target_link_libraries(triangulation ${OpenCV_LIBS})

#add_executable(directSparse direct_sparse.cpp)

add_executable(mono_mapping dense_mapping.cpp)
target_link_libraries(mono_mapping ${OpenCV_LIBS} ${Sophus_LIBRARIES})

