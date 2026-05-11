cmake_minimum_required(VERSION 3.0.0)
project(aie_convert VERSION 0.1.0)

#compile options
add_compile_options(-std=c++11)

IF(NOT DEFINED $ENV{CANN_HOME})
    set(ASCEND_HOME /usr/local/Ascend)
    set(ASCEND_TOOLKIT_HOME ${ASCEND_HOME}/ascend-toolkit/latest)
ENDIF()


add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

LINK_DIRECTORIES(
    $ENV{ASCENDIE_HOME}/lib
    $ENV{CANN_HOME}/x86_64-linux/lib64
    $ENV{CANN_HOME}/aarch64-linux/lib64
)

ADD_EXECUTABLE(aie_convert aie_convert.cpp)

TARGET_INCLUDE_DIRECTORIES(aie_convert PUBLIC
    $ENV{ASCENDIE_HOME}/include
)

TARGET_LINK_LIBRARIES(aie_convert PUBLIC ascendie)
