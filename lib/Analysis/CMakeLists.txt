add_mlir_library(SparsityPropagationAnalysis
  SparsityLattice.cpp
  SparsityPropagationAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSupport
)
