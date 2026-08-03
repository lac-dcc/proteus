add_mlir_library(AddProbeCalls
  AddProbeCalls.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRProbeDialect
  MLIRSupport
)

add_mlir_library(SparsityRewrite
  SparsityRewrite.cpp
  MatmulRewrite.cpp
  Conv2dNchwFchwRewrite.cpp
  DepthConv2dRewrite.cpp
  PoolSumNchwRewrite.cpp
  Utilities.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRRewrite
  MLIRTransformUtils
  MLIRSCFDialect
  MLIRSupport
)
