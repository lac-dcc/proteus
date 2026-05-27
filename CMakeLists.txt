cmake_minimum_required(VERSION 3.20)
project(spa LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

set(CLANG_TIDY_CMD /opt/homebrew/opt/llvm@18/bin/clang-tidy
    --header-filter=^${PROJECT_SOURCE_DIR}/.*
)
foreach(dir ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
    list(APPEND CLANG_TIDY_CMD --extra-arg=-isystem${dir})
endforeach()
set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_CMD})

add_subdirectory(lib)
add_subdirectory(tools)

find_program(CLANG_TIDY_EXE NAMES clang-tidy)
if(CLANG_TIDY_EXE)
    set(TIDY_SOURCES
        ${PROJECT_SOURCE_DIR}/lib/SPAPass.cpp
        ${PROJECT_SOURCE_DIR}/tools/spa-opt/spa-opt.cpp
    )
    set(TIDY_EXTRA_ARGS "")
    foreach(dir ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
        list(APPEND TIDY_EXTRA_ARGS "--extra-arg=-isystem${dir}")
    endforeach()
    add_custom_target(clang-tidy
        COMMAND ${CLANG_TIDY_EXE}
            -p ${CMAKE_BINARY_DIR}
            --header-filter=^${PROJECT_SOURCE_DIR}/.*
            ${TIDY_EXTRA_ARGS}
            ${TIDY_SOURCES}
        COMMENT "Running clang-tidy"
        VERBATIM
    )
endif()
