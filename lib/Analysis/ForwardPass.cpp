#include "Analysis/ForwardPass.h"

#include "Analysis/SeedPass.h"
#include "Analysis/SparsityEngine.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/TypeSwitch.h"

void proteus::ForwardPass::run(mlir::Block *block, SparsityEngine &analysis) {
  for (auto &op : block->getOperations()) {
    analysis.visit<ForwardPass>(op);
  }
}

void proteus::ForwardPass::visit(mlir::Operation &op,
                                 SparsityEngine &analysis) {
  if (op.getNumResults() != 1) {
    return;
  }

  if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantOp>(&op)) {
    SeedPass::seedConstant(constOp, analysis);
    return;
  }

  std::optional<SparsityLattice> lattice;

  if (op.hasAttr("proteus.lattice")) {
    auto attr = op.getAttr("proteus.lattice");
    auto arrayAttr = llvm::cast<mlir::ArrayAttr>(attr);
    lattice = SparsityLattice::fromAttr(arrayAttr);
  } else {
    lattice = SparsityLattice::defaultFromValue(op.getResult(0));
  }

  if (lattice.has_value()) {
    analysis.getState().try_emplace(op.getResult(0), lattice.value());

    mlir::TypeSwitch<mlir::Operation *>(&op)
        .Case<mlir::linalg::MatmulOp, mlir::linalg::AddOp,
              mlir::linalg::MatvecOp, mlir::linalg::VecmatOp,
              mlir::linalg::TransposeOp, mlir::linalg::BatchMatmulOp,
              mlir::linalg::FillOp, mlir::linalg::BroadcastOp,
              mlir::linalg::Conv2DOp, mlir::linalg::Conv2DNchwFchwOp,
              mlir::linalg::Conv2DNhwcHwcfOp, mlir::linalg::PoolingNchwMaxOp,
              mlir::linalg::PoolingNchwSumOp,
              mlir::linalg::DepthwiseConv2DNchwChwOp, mlir::tensor::PadOp,
              mlir::tensor::ConcatOp, mlir::tensor::EmptyOp,
              mlir::tensor::ExpandShapeOp, mlir::tensor::ExtractSliceOp>(
            [&](auto typedOp) -> void { visitOp(typedOp, analysis); })
        .Case<mlir::linalg::AbsOp, mlir::linalg::CeilOp, mlir::linalg::FloorOp,
              mlir::linalg::NegFOp, mlir::linalg::DivOp,
              mlir::linalg::DivUnsignedOp, mlir::linalg::CopyOp,
              mlir::linalg::TanhOp, mlir::linalg::SquareOp,
              mlir::linalg::SqrtOp>(
            [&](auto) -> void { visitPassthroughOp(op, analysis); })
        .Default([](auto) {});
  }
}

void proteus::ForwardPass::visitOp(mlir::linalg::MatmulOp &op,
                                   SparsityEngine &analysis) {
  auto *lhs = analysis.getState(op.getOperand(0));
  auto *rhs = analysis.getState(op.getOperand(1));
  auto *res = analysis.getState(op.getResult(0));

  // We want to keep any sparsity already in the result, this could happen
  // if an attribute is set on a matmul operation
  (*res)[0] &= (*lhs)[0];
  (*res)[1] &= (*rhs)[1];
}

void proteus::ForwardPass::visitOp(mlir::linalg::AddOp &op,
                                   SparsityEngine &analysis) {
  auto *lhs = analysis.getState(op.getOperand(0));
  auto *rhs = analysis.getState(op.getOperand(1));
  auto *res = analysis.getState(op.getResult(0));

  for (std::size_t i = 0; i < res->rank(); i++) {
    // llvm::BitVector does not support binary operations such as lhs & rhs
    // so we have to go with a unary operation instead here.
    llvm::BitVector temp = (*lhs)[i];
    temp |= (*rhs)[i];
    // Again we need to respect any preexisting lattices
    (*res)[i] = temp;
  }
}

void proteus::ForwardPass::visitOp(mlir::linalg::MatvecOp &op,
                                   SparsityEngine &analysis) {
  auto *lhs = analysis.getState(op->getOperand(0));
  auto *rhs = analysis.getState(op->getOperand(1));
  auto *res = analysis.getState(op->getResult(0));

  if ((*rhs)[0].none()) {
    (*res)[0].reset();
  } else {
    (*res)[0] = (*lhs)[0];
  }
}

void proteus::ForwardPass::visitOp(mlir::linalg::VecmatOp &op,
                                   SparsityEngine &analysis) {
  auto *lhs = analysis.getState(op->getOperand(0));
  auto *rhs = analysis.getState(op->getOperand(1));
  auto *res = analysis.getState(op->getResult(0));

  if ((*lhs)[0].none()) {
    (*res)[0].reset();
  } else {
    (*res)[0] = (*rhs)[1];
  }
}

void proteus::ForwardPass::visitOp(mlir::linalg::TransposeOp &op,
                                   SparsityEngine &analysis) {
  auto *input = analysis.getState(op.getOperand(0));
  auto *res = analysis.getState(op->getResult(0));

  llvm::ArrayRef<int64_t> perm = op.getPermutation();
  for (std::size_t i = 0; i < res->rank(); i++) {
    (*res)[i] = (*input)[perm[i]];
  }
}

void proteus::ForwardPass::visitOp(mlir::linalg::BatchMatmulOp &op,
                                   SparsityEngine &analysis) {
  auto *lhs = analysis.getState(op->getOperand(0));
  auto *rhs = analysis.getState(op->getOperand(1));
  auto *res = analysis.getState(op->getResult(0));

  // Transfer sparsity for the batch dimension
  (*res)[0] &= (*lhs)[0];
  (*res)[0] &= (*rhs)[0];
  // Row slice sparsity is maintained from the lhs
  (*res)[1] &= (*lhs)[1];
  // Column slice sparsity in maintained from the rhs
  (*res)[2] &= (*rhs)[2];
}

void proteus::ForwardPass::visitOp(mlir::linalg::FillOp &op,
                                   SparsityEngine &analysis) {
  auto *res = analysis.getState(op.getResult(0));
  mlir::Value cst = op.getInputs()[0];
  auto cstOp = cst.getDefiningOp<mlir::arith::ConstantOp>();

  bool isZero = false;

  // Depending on the type of the constant we have to cast differently
  // In case of FloatType we can get the value of constant through isZero()
  if (mlir::isa<mlir::FloatType>(cst.getType())) {
    isZero = mlir::cast<mlir::FloatAttr>(cstOp.getValue()).getValue().isZero();
    // In case of IntegerType we have to equate
  } else if (mlir::isa<mlir::IntegerType>(cst.getType())) {
    isZero = mlir::cast<mlir::IntegerAttr>(cstOp.getValue()).getInt() == 0;
  }

  // If the constant is zero after all, reset all bits in all ranks
  if (isZero) {
    for (std::size_t i = 0; i < res->rank(); ++i) {
      (*res)[i].reset();
    }
  }
}

void proteus::ForwardPass::visitOp(mlir::linalg::BroadcastOp &op,
                                   SparsityEngine &analysis) {
  auto *input = analysis.getState(op.getInput());
  auto *res = analysis.getState(op->getResult(0));

  // Get the dimensions attribute from the op, these dimensions are essentially
  // all the new dimensions that will be added to the tensor
  auto broadcastDims = op.getDimensions();

  // Broadcast does not allow for the splitting or reshaping of the existing
  // tensor that is being broadcasted, so we will keep a counter of the
  // dimension that we last changed and increment it as we iterate through the
  // ranks of the resulting tensor
  std::size_t inputDim = 0;

  // Iterate through the ranks of our resulting tensor
  for (std::size_t i = 0; i < res->rank(); ++i) {
    // In the case of this dimension not being a part of the dimensions being
    // added by the broadcast continue to next rank
    if (llvm::is_contained(broadcastDims, i)) {
      continue;
    }

    // If we do get to a dimension that existed previously in the input tensor
    // we propagate the sparsity to the current rank's bitvector
    (*res)[i] = (*input)[inputDim++];
  }
}

void proteus::ForwardPass::visitOp(mlir::tensor::PadOp &op,
                                   SparsityEngine &analysis) {
  auto *input = analysis.getState(op.getOperand(0));
  auto *res = analysis.getState(op.getResult());

  // Check all high pads for static padding
  for (auto &pad : op.getMixedLowPad()) {
    // If a padding is not a static, we cannot propagate sparsity
    if (!mlir::getConstantIntValue(pad)) {
      return;
    };
  }

  // Check all high pads for static padding
  for (auto &pad : op.getMixedHighPad()) {
    // If a padding is not a static, we cannot propagate sparsity
    if (!mlir::getConstantIntValue(pad)) {
      return;
    }
  }

  auto pads = op.getMixedLowPad();

  for (std::size_t i = 0; i < res->rank(); ++i) {
    auto lowPad = mlir::getConstantIntValue(pads[i]);

    // Reset all and set as we go through the input lattice for that rank
    (*res)[i].reset();

    for (std::size_t j = 0; j < (*input)[i].size(); j++) {
      // Padded fibers are correct, thus we set by the offset of the padding
      if ((*input)[i][j]) {
        if (lowPad.has_value()) {
          (*res)[i].set(j + lowPad.value());
        }
      }
    }
  }
}

void proteus::ForwardPass::visitOp(mlir::tensor::ConcatOp &op,
                                   SparsityEngine &analysis) {
  auto *res = analysis.getState(op->getResult(0));

  auto concatDim = op.getDim();

  for (std::size_t i = 0; i < res->rank(); ++i) {
    (*res)[i].reset();
  }

  // This offset is needed for the result to be able to concatenate across all
  // tensor operands, so we know at which position we are exactly
  uint64_t offset = 0;

  // The ConcatOp takes an arbitrary number of operands, so we should iterate
  // over the operands this time
  for (std::size_t i = 0; i < op.getNumOperands(); i++) {
    auto *operand = analysis.getState(op.getOperand(i));

    // For each dimension that is not the dimension we are concatenating on
    // we will use the OR operator to ensure that sparsity is propagated when
    // all tensors are sparse on that dimension
    for (std::size_t j = 0; j < res->rank(); j++) {
      if (j == concatDim) {
        // In the case of the concatenation dimension, we just concatenate
        // the bits of that dimension for all tensor operands
        for (std::size_t k = 0; k < (*operand)[j].size(); k++) {
          (*res)[j][k + offset] = (*operand)[j][k];
        }

        offset += (*operand)[j].size();
      } else {
        // For each operand we use the OR operation to make sure that only
        // in the case of all tensors being sparse on that dimension, the
        // sparsity is propagated
        (*res)[j] |= (*operand)[j];
      }
    }
  }
}

void proteus::ForwardPass::visitOp(mlir::tensor::EmptyOp &op,
                                   SparsityEngine &analysis) {}

void proteus::ForwardPass::visitOp(mlir::linalg::Conv2DOp &op,
                                   SparsityEngine &analysis) {
  auto *input = analysis.getState(op.getOperand(0));
  auto *res = analysis.getState(op.getResult(0));

  auto filterType =
      llvm::cast<mlir::RankedTensorType>(op.getOperand(1).getType());

  // Iterate through each rank of the resulting tensor
  for (std::size_t i = 0; i < res->rank(); i++) {
    // Walk through each fiber on that dimension
    for (std::size_t fiber = 0; fiber < (*res)[i].size(); fiber++) {
      bool allSparse = true;
      // Check whether there are f_fiber contiguous zeros, if there are the
      // result of the convolution for the fiber is also sparse
      for (int64_t fFiber = 0; fFiber < filterType.getDimSize(i) && allSparse;
           fFiber++) {
        //  If not falsify the allSparse flag and continue with the rest of the
        //  fibers in the result
        if ((*input)[i][fiber + fFiber]) {
          allSparse = false;
        }
      }

      // If the allSparse flag is true by the end of the above iteration,
      // this is where we make the resulting fiber sparse
      if (allSparse) {
        (*res)[i].reset(fiber);
      }
    }
  }
}

void proteus::ForwardPass::visitOp(mlir::linalg::Conv2DNchwFchwOp &op,
                                   SparsityEngine &analysis) {
  auto *input = analysis.getState(op.getOperand(0));
  auto *filter = analysis.getState(op.getOperand(1));
  auto *res = analysis.getState(op.getResult(0));

  auto filterType =
      llvm::cast<mlir::RankedTensorType>(op.getOperand(1).getType());

  llvm::SmallVector<int64_t, 2> strides(op.getStrides().getValues<int64_t>());
  llvm::SmallVector<int64_t, 2> dilations(
      op.getDilations().getValues<int64_t>());

  // Sparsity for batches and channels is propagated in a passthrough fashion
  (*res)[0] &= (*input)[0];
  (*res)[1] &= (*filter)[0];

  // We follow the same logic as we did we the simple conv2d case with the
  // spatial dimensions
  for (uint64_t dim = 0; dim < 2; dim++) {
    int64_t kernelSize = filterType.getDimSize(dim + 2);
    int64_t stride = strides[dim];
    int64_t dilation = dilations[dim];

    for (std::size_t fiber = 0; fiber < (*res)[dim + 2].size(); fiber++) {
      bool allSparse = true;
      for (int64_t fFiber = 0; fFiber < kernelSize && allSparse; fFiber++) {
        // Only this time we need to account for specific stride and dilation
        if ((*input)[dim + 2][(fiber * stride) + (fFiber * dilation)]) {
          allSparse = false;
        }
      }

      if (allSparse) {
        (*res)[dim + 2].reset(fiber);
      }
    }
  }
}

void proteus::ForwardPass::visitOp(mlir::linalg::Conv2DNhwcHwcfOp &op,
                                   SparsityEngine &analysis) {
  auto *input = analysis.getState(op.getOperand(0));
  auto *filter = analysis.getState(op.getOperand(1));
  auto *res = analysis.getState(op.getResult(0));

  auto filterType =
      llvm::cast<mlir::RankedTensorType>(op.getOperand(1).getType());

  llvm::SmallVector<int64_t, 2> strides(op.getStrides().getValues<int64_t>());
  llvm::SmallVector<int64_t, 2> dilations(
      op.getDilations().getValues<int64_t>());

  // Sparsity for batches and channels is propagated in a passthrough fashion
  (*res)[0] &= (*input)[0];
  (*res)[3] &= (*filter)[3];

  // Same idea as above with the Conv2DNchwFchwOp format, just dimensions are
  // different this time
  for (uint64_t dim = 0; dim < 2; dim++) {
    int64_t kernelSize = filterType.getDimSize(dim);
    int64_t stride = strides[dim];
    int64_t dilation = dilations[dim];

    for (std::size_t fiber = 0; fiber < (*res)[dim + 1].size(); fiber++) {
      bool allSparse = true;
      for (int64_t fFiber = 0; fFiber < kernelSize && allSparse; fFiber++) {
        if ((*input)[dim + 1][(fiber * stride) + (fFiber * dilation)]) {
          allSparse = false;
        }
      }

      if (allSparse) {
        (*res)[dim + 1].reset(fiber);
      }
    }
  }
}

void proteus::ForwardPass::visitOp(mlir::linalg::DepthwiseConv2DNchwChwOp &op,
                                   SparsityEngine &analysis) {
  auto *input = analysis.getState(op.getOperand(0));
  auto *filter = analysis.getState(op.getOperand(1));
  auto *res = analysis.getState(op.getResult(0));

  auto filterType =
      llvm::cast<mlir::RankedTensorType>(op.getOperand(1).getType());

  llvm::SmallVector<int64_t, 2> strides(op.getStrides().getValues<int64_t>());
  llvm::SmallVector<int64_t, 2> dilations(
      op.getDilations().getValues<int64_t>());

  (*res)[0] &= (*input)[0];
  // Channels here convolve independently and so the channel dimension in the
  // depthwise case will depend on the sparsity of both the input and the filter
  (*res)[1] &= (*input)[1];
  (*res)[1] &= (*filter)[0];

  for (uint64_t dim = 0; dim < 2; dim++) {
    int64_t kernelSize = filterType.getDimSize(dim + 1);
    int64_t stride = strides[dim];
    int64_t dilation = dilations[dim];

    for (std::size_t fiber = 0; fiber < (*res)[dim + 2].size(); fiber++) {
      bool allSparse = true;
      for (int64_t fFiber = 0; fFiber < kernelSize && allSparse; fFiber++) {
        if ((*input)[dim + 2][(fiber * stride) + (fFiber * dilation)]) {
          allSparse = false;
        }
      }

      if (allSparse) {
        (*res)[dim + 2].reset(fiber);
      }
    }
  }
}

void proteus::ForwardPass::visitOp(mlir::linalg::PoolingNchwMaxOp &op,
                                   SparsityEngine &analysis) {
  auto *input = analysis.getState(op.getOperand(0));
  auto *res = analysis.getState(op.getResult(0));

  auto kernelType =
      llvm::cast<mlir::RankedTensorType>(op.getOperand(1).getType());

  llvm::SmallVector<int64_t, 2> strides(op.getStrides().getValues<int64_t>());
  llvm::SmallVector<int64_t, 2> dilations(
      op.getDilations().getValues<int64_t>());

  // Same logic for pooling as with convolutions in 2D
  (*res)[0] &= (*input)[0];
  (*res)[1] &= (*input)[1];

  // Same logic for pooling as with convolutions in 2D
  for (uint64_t dim = 0; dim < 2; dim++) {
    int64_t kernelSize = kernelType.getDimSize(dim);
    int64_t stride = strides[dim];
    int64_t dilation = dilations[dim];

    for (std::size_t fiber = 0; fiber < (*res)[dim + 2].size(); fiber++) {
      bool allSparse = true;
      for (int64_t fFiber = 0; fFiber < kernelSize && allSparse; fFiber++) {
        if ((*input)[dim + 2][(fiber * stride) + (fFiber * dilation)]) {
          allSparse = false;
        }
      }

      if (allSparse) {
        (*res)[dim + 2].reset(fiber);
      }
    }
  }
}

void proteus::ForwardPass::visitOp(mlir::linalg::PoolingNchwSumOp &op,
                                   SparsityEngine &analysis) {
  auto *input = analysis.getState(op.getOperand(0));
  auto *res = analysis.getState(op.getResult(0));

  auto kernelType =
      llvm::cast<mlir::RankedTensorType>(op.getOperand(1).getType());

  llvm::SmallVector<int64_t, 2> strides(op.getStrides().getValues<int64_t>());
  llvm::SmallVector<int64_t, 2> dilations(
      op.getDilations().getValues<int64_t>());

  // Again, same logic for pooling as with convolutions in 2D
  (*res)[0] &= (*input)[0];
  (*res)[1] &= (*input)[1];

  // Again, same logic for pooling as with convolutions in 2D
  for (uint64_t dim = 0; dim < 2; dim++) {
    int64_t kernelSize = kernelType.getDimSize(dim);
    int64_t stride = strides[dim];
    int64_t dilation = dilations[dim];

    for (std::size_t fiber = 0; fiber < (*res)[dim + 2].size(); fiber++) {
      bool allSparse = true;
      for (int64_t fFiber = 0; fFiber < kernelSize && allSparse; fFiber++) {
        if ((*input)[dim + 2][(fiber * stride) + (fFiber * dilation)]) {
          allSparse = false;
        }
      }

      if (allSparse) {
        (*res)[dim + 2].reset(fiber);
      }
    }
  }
}

void proteus::ForwardPass::visitOp(mlir::tensor::ExpandShapeOp &op,
                                   SparsityEngine &analysis) {
  auto *input = analysis.getState(op.getOperand(0));
  auto *res = analysis.getState(op->getResult(0));

  auto inputType =
      mlir::cast<mlir::RankedTensorType>(op.getOperand(0).getType());
  auto resType = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // Reassociation is the MLIR attribute attached to the tensor.expand_shape op
  // that holds information about how the expansion of ranks is done for the
  // resulting tensor
  auto reassociation = op.getReassociationIndices();

  for (auto [srcDim, group] : llvm::enumerate(reassociation)) {
    int64_t srcDimSize = inputType.getDimSize(srcDim);

    // For each dimension in the input we need to know how to iterate through
    // the dimensions of the reshaped result tensor through row major flattening
    // using flat indeces. For example: (i, j) lives at a flat index of i *
    // coefficient[0] + j and this can be done recursively across N dimensions.
    // The code below extracts those coefficients so we can later identify the
    // position of a bit in the resulting tensor
    llvm::SmallVector<int64_t> coefficients(group.size());
    // The last coefficient for any rank is always one
    coefficients.back() = 1;
    // Now we go through each position backwards, one at a time calculating the
    // next coefficient based on the dimension and coefficient of the previous
    // position
    for (int p = static_cast<int>(group.size()) - 2; p >= 0; --p) {
      coefficients[p] = coefficients[p + 1] * resType.getDimSize(group[p + 1]);
    }

    for (auto [p, outDim] : llvm::enumerate(group)) {
      int64_t outSize = resType.getDimSize(outDim);

      // We create a new all sparse bitvector that we will explicitly compute
      llvm::BitVector computed(outSize, false);
      for (int64_t i = 0; i < srcDimSize; ++i) {
        // If there exists a bit that is set in the input
        if ((*input)[srcDim][i]) {
          // Set the corresponding fiber in the resulting tensor
          // by using the domain decomposition coefficients
          // calculated above
          int64_t j = (i / coefficients[p]) % outSize;
          computed.set(j);
        }
      }

      // Have the computed bitvector included in the resulting tensor
      (*res)[outDim] &= computed;
    }
  }
}

void proteus::ForwardPass::visitOp(mlir::tensor::ExtractSliceOp &op,
                                   SparsityEngine &analysis) {
  auto *input = analysis.getState(op.getOperand(0));
  auto *res = analysis.getState(op->getResult(0));

  // MLIR attributes attached to the tensor.extract_slice op that hold
  // information on where each slice begins, the size of the slice, and the size
  // of the stride if the accesses are strided
  auto offsets = op.getMixedOffsets();
  auto sizes = op.getMixedSizes();
  auto strides = op.getMixedStrides();

  // Helper lambda function that check that are items in the above attributes
  // are constants If the case of dynamic attributes, we cannot propagate
  // sparsity
  auto checkConst = [](const auto &vector) {
    for (auto &item : vector) {
      if (!mlir::getConstantIntValue(item).has_value()) {
        return;
      }
    }
  };

  checkConst(offsets);
  checkConst(sizes);
  checkConst(strides);

  uint64_t resDim = 0;
  for (uint64_t srcDim = 0; srcDim < offsets.size(); srcDim++) {
    auto size = mlir::getConstantIntValue(sizes[srcDim]);

    // In case that the size is equal to 1, the source dimension is reduced and
    // there exists no corresponding result dimension, which means we should
    // skip
    if (size.has_value() && size.value() == 1) {
      continue;
    }

    int64_t offset = mlir::getConstantIntValue(offsets[srcDim]).value();
    int64_t stride = mlir::getConstantIntValue(strides[srcDim]).value();

    // Same as with what we did with tensor.expand_shape, we will create
    // an initial bitvector that we will explicitly construct from the sparsity
    // information of the corresponding fibers in the input tensor
    llvm::BitVector computed(size.value(), false);
    for (int64_t j = 0; j < size.value(); ++j) {
      // This formula gives as the fiber in the input tensor for the current
      // dimension
      int64_t srcFiber = offset + (j * stride);
      // If that fiber is set, then the corresponding fiber in the result tensor
      // should also be set
      if ((*input)[srcDim][srcFiber]) {
        computed.set(j);
      }
    }

    // Pass the computed bitvector to the result
    (*res)[resDim] &= computed;
    ++resDim;
  }
}

void proteus::ForwardPass::visitPassthroughOp(mlir::Operation &op,
                                              SparsityEngine &analysis) {
  auto *operand = analysis.getState(op.getOperand(0));
  auto *res = analysis.getState(op.getOpResult(0));

  for (std::size_t i = 0; i < res->rank(); i++) {
    (*res)[i] = (*operand)[i];
  }
}
