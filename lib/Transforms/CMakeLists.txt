add_mlir_library(AddProbeCalls
  AddProbeCalls.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRProbeDialect
  MLIRSupport
)
