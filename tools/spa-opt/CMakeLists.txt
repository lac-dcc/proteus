add_llvm_executable(spa-opt
  spa-opt.cpp
)

set_target_properties(spa-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_link_libraries(spa-opt PRIVATE
  SparsityPropagationAnalysis
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIROptLib
  MLIRPass
  MLIRIR
)
