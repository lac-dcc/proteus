add_llvm_executable(proteus-opt
  proteus-opt.cpp
)

set_target_properties(proteus-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_link_libraries(proteus-opt PRIVATE
  SparsityPropagationAnalysis
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIROptLib
  MLIRPass
  MLIRIR
)
