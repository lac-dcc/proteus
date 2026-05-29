add_mlir_library(SparsityPropagationAnalysis
  SparsityPropagationAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSupport
)
