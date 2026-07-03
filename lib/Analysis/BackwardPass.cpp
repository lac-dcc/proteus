#include "Analysis/BackwardPass.h"

#include "Analysis/SparsityEngine.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/TypeSwitch.h"

void proteus::BackwardPass::run(mlir::Block *block, SparsityEngine &analysis) {
  auto worklist = getWorklist(block);

  while (!worklist.empty()) {
    analysis.setCandidateValue(worklist.pop_back_val());

    auto *current = analysis.getState(analysis.getCandidateValue());

    std::optional<SparsityLattice> joined;
    for (mlir::Operation *user : analysis.getCandidateValue().getUsers()) {
      auto candidate = visit(*user, analysis);
      if (!candidate) {
        continue;
      }
      joined = joined ? SparsityLattice::join(*joined, *candidate) : candidate;
    }

    if (!joined) {
      continue;
    }

    auto merged = SparsityLattice::meet(*current, *joined);
    if (merged != *current) {
      *current = merged;

      for (mlir::Operation *user : analysis.getCandidateValue().getUsers()) {
        for (mlir::Value operand : user->getOperands()) {
          worklist.push_back(operand);
        }
      }
    }
  }
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visit(mlir::Operation &op, SparsityEngine &analysis) {
  if (op.getNumResults() != 1) {
    return *analysis.getState(analysis.getCandidateValue());
  }

  // Based on the mlir operation we dispatch the appropriate transfer function
  return mlir::TypeSwitch<mlir::Operation *,
                          std::optional<proteus::SparsityLattice>>(&op)
      .Case<mlir::linalg::MatmulOp, mlir::linalg::AddOp, mlir::linalg::MatvecOp,
            mlir::linalg::VecmatOp, mlir::linalg::TransposeOp,
            mlir::linalg::BatchMatmulOp, mlir::linalg::FillOp,
            mlir::linalg::BroadcastOp, mlir::linalg::Conv2DOp,
            mlir::linalg::Conv2DNchwFchwOp, mlir::linalg::Conv2DNhwcHwcfOp,
            mlir::linalg::PoolingNchwMaxOp, mlir::linalg::PoolingNchwSumOp,
            mlir::linalg::DepthwiseConv2DNchwChwOp, mlir::tensor::PadOp,
            mlir::tensor::ConcatOp, mlir::tensor::EmptyOp,
            mlir::tensor::ExpandShapeOp, mlir::tensor::ExtractSliceOp,
            mlir::tensor::CollapseShapeOp>(
          [&](auto typedOp) -> std::optional<proteus::SparsityLattice> {
            return visitOp(typedOp, analysis);
          })
      .Case<mlir::linalg::AbsOp, mlir::linalg::CeilOp, mlir::linalg::FloorOp,
            mlir::linalg::NegFOp, mlir::linalg::DivOp,
            mlir::linalg::DivUnsignedOp, mlir::linalg::CopyOp,
            mlir::linalg::TanhOp, mlir::linalg::SquareOp, mlir::linalg::SqrtOp>(
          [&](auto) -> std::optional<SparsityLattice> {
            return visitPassthroughOp(op, analysis);
          })
      .Default(
          [&](mlir::Operation *op) -> std::optional<proteus::SparsityLattice> {
            if (op->getNumResults() == 0) {
              return std::nullopt;
            }

            return *analysis.getState(analysis.getCandidateValue());
          });
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::MatmulOp &op,
                               SparsityEngine &analysis) {
  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::AddOp &op,
                               SparsityEngine &analysis) {
  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::MatvecOp &op,
                               SparsityEngine &analysis) {
  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::VecmatOp &op,
                               SparsityEngine &analysis) {
  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::TransposeOp &op,
                               SparsityEngine &analysis) {
  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::BatchMatmulOp &op,
                               SparsityEngine &analysis) {
  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::FillOp &op,
                               SparsityEngine &analysis) {
  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::BroadcastOp &op,
                               SparsityEngine &analysis) {
  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::tensor::PadOp &op,
                               SparsityEngine &analysis) {
  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::tensor::ConcatOp &op,
                               SparsityEngine &analysis) {
  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::tensor::EmptyOp &op,
                               SparsityEngine &analysis) {
  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::Conv2DOp &op,
                               SparsityEngine &analysis) {
  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::Conv2DNchwFchwOp &op,
                               SparsityEngine &analysis) {
  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::Conv2DNhwcHwcfOp &op,
                               SparsityEngine &analysis) {
  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::DepthwiseConv2DNchwChwOp &op,
                               SparsityEngine &analysis) {
  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::PoolingNchwMaxOp &op,
                               SparsityEngine &analysis) {
  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::PoolingNchwSumOp &op,
                               SparsityEngine &analysis) {
  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::tensor::ExpandShapeOp &op,
                               SparsityEngine &analysis) {
  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::tensor::ExtractSliceOp &op,
                               SparsityEngine &analysis) {
  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::tensor::CollapseShapeOp &op,
                               SparsityEngine &analysis) {
  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitPassthroughOp(mlir::Operation &op,
                                          SparsityEngine &analysis) {
  return std::nullopt;
}

llvm::SmallVector<mlir::Value>
proteus::BackwardPass::getWorklist(mlir::Block *block) {
  llvm::SmallVector<mlir::Value> worklist;

  for (auto &op : block->getOperations()) {
    for (auto operand : op.getOperands()) {
      worklist.push_back(operand);
    }
  }

  return worklist;
}
