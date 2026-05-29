add_llvm_executable(spa-opt
  spa-opt.cpp
)

target_link_libraries(spa-opt PRIVATE
  SPAPass
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIROptLib
  MLIRPass
  MLIRIR
)
