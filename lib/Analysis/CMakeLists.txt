add_mlir_library(SparsityPropagationAnalysis
  SparsityLattice.cpp
  SparsityPropagationAnalysis.cpp
  SparsityEngine.cpp
  Utilities.cpp
  SeedPass.cpp
  ForwardPass.cpp
  LateralPass.cpp
  BackwardPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRSupport
)
