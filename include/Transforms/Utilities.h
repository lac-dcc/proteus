#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace proteus {
mlir::Value
checkIterDensity(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iter,
                 llvm::ArrayRef<std::pair<int64_t, int64_t>> ranges);
}
