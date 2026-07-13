add_library(ProteusProbeRuntime SHARED
  ProbeRuntime.cpp
)

set_target_properties(ProteusProbeRuntime PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

target_link_libraries(ProteusProbeRuntime PRIVATE SparsityPropagationAnalysis LLVMSupport)
