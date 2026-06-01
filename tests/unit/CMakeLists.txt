find_package(GTest REQUIRED)

add_executable(SparsityLatticeTests SparsityLatticeTest.cpp)

target_link_libraries(SparsityLatticeTests
    PRIVATE
    GTest::GTest
    GTest::Main
    SparsityPropagationAnalysis
    MLIRIR
)

# GTest_DIR is <prefix>/lib/cmake/GTest — two levels up is the lib dir.
get_filename_component(_gtest_rpath "${GTest_DIR}/../.." ABSOLUTE)
set_target_properties(SparsityLatticeTests PROPERTIES BUILD_RPATH "${_gtest_rpath}")

enable_testing()
add_test(NAME SparsityLatticeTests COMMAND SparsityLatticeTests)
