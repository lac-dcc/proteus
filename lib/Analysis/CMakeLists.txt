add_mlir_library(SparsityPropagationAnalysis
  SparsityLattice.cpp
  SparsityPropagationAnalysis.cpp
  SparsityAnalysisDriver.cpp
  Utilities.cpp
  ForwardPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRSupport
)
