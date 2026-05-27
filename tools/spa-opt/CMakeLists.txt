get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(spa-opt
  spa-opt.cpp
)

target_link_libraries(spa-opt PRIVATE
  SPAPass
  ${dialect_libs}
  ${extension_libs}
  ${conversion_libs}
  MLIROptLib
  MLIRPass
  MLIRIR
)
