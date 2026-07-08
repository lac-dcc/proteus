#include "Analysis/BackwardPass.h"

#include "Analysis/SparsityEngine.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
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

      // TODO: We probably don't need this because changes in operands are
      // completely independent in the backward pass and also there is no loops
      // or if statements currenlty in the IR that we are working with for
      // (mlir::Operation *user : analysis.getCandidateValue().getUsers()) {
      //   for (mlir::Value operand : user->getOperands()) {
      //     if (mlir::isa<mlir::RankedTensorType>(operand.getType())) {
      //       worklist.push_back(operand);
      //     }
      //   }
      // }
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
            mlir::linalg::BatchMatmulOp, mlir::linalg::BroadcastOp,
            mlir::linalg::Conv2DOp, mlir::linalg::Conv2DNchwFchwOp,
            mlir::linalg::Conv2DNhwcHwcfOp, mlir::linalg::PoolingNchwMaxOp,
            mlir::linalg::PoolingNchwSumOp,
            mlir::linalg::DepthwiseConv2DNchwChwOp, mlir::tensor::PadOp,
            mlir::tensor::ConcatOp, mlir::tensor::ExpandShapeOp,
            mlir::tensor::ExtractSliceOp, mlir::tensor::CollapseShapeOp>(
          [&](auto typedOp) -> std::optional<proteus::SparsityLattice> {
            return visitOp(typedOp, analysis);
          })
      .Case<mlir::linalg::AbsOp, mlir::linalg::CeilOp, mlir::linalg::FloorOp,
            mlir::linalg::NegFOp, mlir::linalg::DivOp,
            mlir::linalg::DivUnsignedOp, mlir::linalg::CopyOp,
            mlir::linalg::TanhOp, mlir::linalg::SquareOp, mlir::linalg::SqrtOp>(
          [&](auto) -> SparsityLattice {
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

  auto *res = analysis.getState(op.getResult(0));
  auto candidate = *analysis.getState(analysis.getCandidateValue());

  if (analysis.getCandidateValue() == op.getOperand(0)) {
    candidate[0] &= (*res)[0];
  }
  if (analysis.getCandidateValue() == op.getOperand(1)) {
    candidate[1] &= (*res)[1];
  }

  return candidate;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::AddOp &op,
                               SparsityEngine &analysis) {

  auto *res = analysis.getState(op.getResult(0));
  auto candidate = *analysis.getState(analysis.getCandidateValue());

  for (std::size_t i = 0; i < candidate.rank(); i++) {
    candidate[i] &= (*res)[i];
  }

  return candidate;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::MatvecOp &op,
                               SparsityEngine &analysis) {

  if (analysis.getCandidateValue() != op.getOperand(0)) {
    return std::nullopt;
  }

  auto *res = analysis.getState(op.getResult(0));
  auto candidate = *analysis.getState(analysis.getCandidateValue());
  candidate[0] &= (*res)[0];

  return candidate;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::VecmatOp &op,
                               SparsityEngine &analysis) {

  if (analysis.getCandidateValue() != op.getOperand(1)) {
    return std::nullopt;
  }

  auto *res = analysis.getState(op.getResult(0));
  auto candidate = *analysis.getState(analysis.getCandidateValue());
  candidate[1] &= (*res)[0];

  return candidate;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::TransposeOp &op,
                               SparsityEngine &analysis) {

  auto *res = analysis.getState(op->getResult(0));
  auto candidate = *analysis.getState(analysis.getCandidateValue());

  llvm::ArrayRef<int64_t> perm = op.getPermutation();
  for (std::size_t i = 0; i < candidate.rank(); i++) {
    candidate[perm[i]] &= (*res)[i];
  }

  return candidate;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::BatchMatmulOp &op,
                               SparsityEngine &analysis) {
  auto *res = analysis.getState(op.getResult(0));
  auto candidate = *analysis.getState(analysis.getCandidateValue());

  if (analysis.getCandidateValue() == op.getOperand(0)) {
    candidate[1] &= (*res)[1];
  }
  if (analysis.getCandidateValue() == op.getOperand(1)) {
    candidate[2] &= (*res)[2];
  }

  return candidate;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::BroadcastOp &op,
                               SparsityEngine &analysis) {

  auto *res = analysis.getState(op->getResult(0));
  auto candidate = *analysis.getState(analysis.getCandidateValue());

  auto broadcastDims = op.getDimensions();
  std::size_t inputDim = 0;
  for (std::size_t i = 0; i < res->rank(); ++i) {
    if (llvm::is_contained(broadcastDims, i)) {
      continue;
    }
    candidate[inputDim++] &= (*res)[i];
  }

  return candidate;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::tensor::PadOp &op,
                               SparsityEngine &analysis) {
  return *analysis.getState(analysis.getCandidateValue());
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::tensor::ConcatOp &op,
                               SparsityEngine &analysis) {
  return *analysis.getState(analysis.getCandidateValue());
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::Conv2DOp &op,
                               SparsityEngine &analysis) {
  if (analysis.getCandidateValue() != op.getOperand(0)) {
    return *analysis.getState(analysis.getCandidateValue());
  }

  auto *res = analysis.getState(op.getResult(0));
  auto candidate = *analysis.getState(analysis.getCandidateValue());

  auto filterType =
      llvm::cast<mlir::RankedTensorType>(op.getOperand(1).getType());

  // For each spatial dimension
  for (std::size_t i = 0; i < 2; i++) {
    // Create a new bitvector to operate on, that will be ANDed to the
    // corresponding candidate's dimension
    llvm::BitVector vec(candidate[i].size(), true);
    // For each result fiber in that spatial dimension
    for (std::size_t fiber = 0; fiber < (*res)[i].size(); fiber++) {
      // If that fiber is dense move on, we won't change any of the bits in vec
      // so move on
      if ((*res)[i][fiber]) {
        continue;
      }

      // In the case where the fiber is sparse, we will sequentially make sparse
      // al the bits in that filter window in vec
      for (int64_t fFiber = 0; fFiber < filterType.getDimSize(i); fFiber++) {
        if (fiber + fFiber < candidate[i].size()) {
          vec.reset(fiber + fFiber);
        }
      }
    }

    // Append vec to the candidate and move on to the next spatial dimension
    candidate[i] &= vec;
  }

  return candidate;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::Conv2DNchwFchwOp &op,
                               SparsityEngine &analysis) {
  if (analysis.getCandidateValue() != op.getOperand(0)) {
    return *analysis.getState(analysis.getCandidateValue());
  }

  auto *res = analysis.getState(op.getResult(0));
  auto candidate = *analysis.getState(analysis.getCandidateValue());

  auto filterType =
      llvm::cast<mlir::RankedTensorType>(op.getOperand(1).getType());

  llvm::SmallVector<int64_t, 2> strides(op.getStrides().getValues<int64_t>());
  llvm::SmallVector<int64_t, 2> dilations(
      op.getDilations().getValues<int64_t>());

  // For each spatial dimension
  for (uint64_t dim = 0; dim < 2; dim++) {
    auto stride = strides[dim];
    auto dilation = dilations[dim];

    // Create a new bitvector to operate on, that will be ANDed to the
    // corresponding candidate's dimension
    llvm::BitVector vec(candidate[dim + 2].size(), true);
    // For each result fiber in that spatial dimension
    for (std::size_t fiber = 0; fiber < (*res)[dim + 2].size(); fiber++) {
      // If that fiber is dense move on, we won't change any of the bits in vec
      if ((*res)[dim + 2][fiber]) {
        continue;
      }

      // In the case where the fiber is sparse, we will sequentially make
      // sparse all the bits in that filter window in vec
      for (int64_t fFiber = 0; fFiber < filterType.getDimSize(dim + 2);
           fFiber++) {
        std::size_t pos = (fiber * stride) + (fFiber * dilation);
        if (pos < candidate[dim + 2].size()) {
          vec.reset(pos);
        }
      }
    }

    // Append vec to the candidate and move on to the next spatial dimension
    candidate[dim + 2] &= vec;
  }

  return candidate;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::Conv2DNhwcHwcfOp &op,
                               SparsityEngine &analysis) {
  if (analysis.getCandidateValue() != op.getOperand(0)) {
    return *analysis.getState(analysis.getCandidateValue());
  }

  auto *res = analysis.getState(op.getResult(0));
  auto candidate = *analysis.getState(analysis.getCandidateValue());

  auto filterType =
      llvm::cast<mlir::RankedTensorType>(op.getOperand(1).getType());

  llvm::SmallVector<int64_t, 2> strides(op.getStrides().getValues<int64_t>());
  llvm::SmallVector<int64_t, 2> dilations(
      op.getDilations().getValues<int64_t>());

  // For each spatial dimension
  for (uint64_t dim = 0; dim < 2; dim++) {
    auto stride = strides[dim];
    auto dilation = dilations[dim];

    // Create a new bitvector to operate on, that will be ANDed to the
    // corresponding candidate's dimension
    llvm::BitVector vec(candidate[dim + 1].size(), true);
    // For each result fiber in that spatial dimension
    for (std::size_t fiber = 0; fiber < (*res)[dim + 1].size(); fiber++) {
      // If that fiber is dense move on, we won't change any of the bits in vec
      if ((*res)[dim + 1][fiber]) {
        continue;
      }

      // In the case where the fiber is sparse, we will sequentially make
      // sparse all the bits in that filter window in vec
      for (int64_t fFiber = 0; fFiber < filterType.getDimSize(dim); fFiber++) {
        std::size_t pos = (fiber * stride) + (fFiber * dilation);
        if (pos < candidate[dim + 1].size()) {
          vec.reset(pos);
        }
      }
    }

    // Append vec to the candidate and move on to the next spatial dimension
    candidate[dim + 1] &= vec;
  }

  return candidate;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::DepthwiseConv2DNchwChwOp &op,
                               SparsityEngine &analysis) {
  if (analysis.getCandidateValue() != op.getOperand(0)) {
    return *analysis.getState(analysis.getCandidateValue());
  }

  auto *res = analysis.getState(op.getResult(0));
  auto candidate = *analysis.getState(analysis.getCandidateValue());

  auto filterType =
      llvm::cast<mlir::RankedTensorType>(op.getOperand(1).getType());

  llvm::SmallVector<int64_t, 2> strides(op.getStrides().getValues<int64_t>());
  llvm::SmallVector<int64_t, 2> dilations(
      op.getDilations().getValues<int64_t>());

  for (uint64_t dim = 0; dim < 2; dim++) {
    auto stride = strides[dim];
    auto dilation = dilations[dim];

    llvm::BitVector vec(candidate[dim + 2].size(), true);
    for (std::size_t fiber = 0; fiber < (*res)[dim + 2].size(); fiber++) {
      if ((*res)[dim + 2][fiber]) {
        continue;
      }

      for (int64_t fFiber = 0; fFiber < filterType.getDimSize(dim + 1);
           fFiber++) {
        std::size_t pos = (fiber * stride) + (fFiber * dilation);
        if (pos < candidate[dim + 2].size()) {
          vec.reset(pos);
        }
      }
    }

    candidate[dim + 2] &= vec;
  }

  return candidate;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::PoolingNchwMaxOp &op,
                               SparsityEngine &analysis) {
  if (analysis.getCandidateValue() != op.getOperand(0)) {
    return *analysis.getState(analysis.getCandidateValue());
  }

  auto *res = analysis.getState(op.getResult(0));
  auto candidate = *analysis.getState(analysis.getCandidateValue());

  auto kernelType =
      llvm::cast<mlir::RankedTensorType>(op.getOperand(1).getType());

  llvm::SmallVector<int64_t, 2> strides(op.getStrides().getValues<int64_t>());
  llvm::SmallVector<int64_t, 2> dilations(
      op.getDilations().getValues<int64_t>());

  for (uint64_t dim = 0; dim < 2; dim++) {
    auto stride = strides[dim];
    auto dilation = dilations[dim];

    llvm::BitVector vec(candidate[dim + 2].size(), true);
    for (std::size_t fiber = 0; fiber < (*res)[dim + 2].size(); fiber++) {
      if ((*res)[dim + 2][fiber]) {
        continue;
      }

      for (int64_t fFiber = 0; fFiber < kernelType.getDimSize(dim); fFiber++) {
        std::size_t pos = (fiber * stride) + (fFiber * dilation);
        if (pos < candidate[dim + 2].size()) {
          vec.reset(pos);
        }
      }
    }

    candidate[dim + 2] &= vec;
  }

  return candidate;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::linalg::PoolingNchwSumOp &op,
                               SparsityEngine &analysis) {
  if (analysis.getCandidateValue() != op.getOperand(0)) {
    return *analysis.getState(analysis.getCandidateValue());
  }

  auto *res = analysis.getState(op.getResult(0));
  auto candidate = *analysis.getState(analysis.getCandidateValue());

  auto kernelType =
      llvm::cast<mlir::RankedTensorType>(op.getOperand(1).getType());

  llvm::SmallVector<int64_t, 2> strides(op.getStrides().getValues<int64_t>());
  llvm::SmallVector<int64_t, 2> dilations(
      op.getDilations().getValues<int64_t>());

  for (uint64_t dim = 0; dim < 2; dim++) {
    auto stride = strides[dim];
    auto dilation = dilations[dim];

    llvm::BitVector vec(candidate[dim + 2].size(), true);
    for (std::size_t fiber = 0; fiber < (*res)[dim + 2].size(); fiber++) {
      if ((*res)[dim + 2][fiber]) {
        continue;
      }

      for (int64_t fFiber = 0; fFiber < kernelType.getDimSize(dim); fFiber++) {
        std::size_t pos = (fiber * stride) + (fFiber * dilation);
        if (pos < candidate[dim + 2].size()) {
          vec.reset(pos);
        }
      }
    }

    candidate[dim + 2] &= vec;
  }

  return candidate;
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::tensor::ExpandShapeOp &op,
                               SparsityEngine &analysis) {
  return *analysis.getState(analysis.getCandidateValue());
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::tensor::ExtractSliceOp &op,
                               SparsityEngine &analysis) {
  return *analysis.getState(analysis.getCandidateValue());
}

std::optional<proteus::SparsityLattice>
proteus::BackwardPass::visitOp(mlir::tensor::CollapseShapeOp &op,
                               SparsityEngine &analysis) {
  return *analysis.getState(analysis.getCandidateValue());
}

proteus::SparsityLattice
proteus::BackwardPass::visitPassthroughOp(mlir::Operation &op,
                                          SparsityEngine &analysis) {

  auto *src = analysis.getState(op.getOperand(0));
  auto *res = analysis.getState(op.getResult(0));

  SparsityLattice candidate = *src;

  for (std::size_t i = 0; i < res->rank(); i++) {
    candidate[i] &= (*res)[i];
  }

  return candidate;
}

llvm::SmallVector<mlir::Value>
proteus::BackwardPass::getWorklist(mlir::Block *block) {
  llvm::SmallVector<mlir::Value> worklist;

  for (auto &op : block->getOperations()) {
    if (auto dpsOp = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(op)) {
      for (auto operand : dpsOp.getDpsInputs()) {
        if (mlir::isa<mlir::RankedTensorType>(operand.getType())) {
          worklist.push_back(operand);
        }
      }
    }
  }

  return worklist;
}
