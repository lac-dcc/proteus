add_mlir_library(SparsityPropagationAnalysis
  SparsityLattice.cpp
  SparsityPropagationAnalysis.cpp
  SparsityAnalysis.cpp
  Utilities.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRSupport
)
