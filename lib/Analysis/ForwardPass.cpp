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

  // We first seed the constants with their respective lattices, which we should
  // be able to infer at compile time, and return
  if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantOp>(&op)) {
    SeedPass::seedConstant(constOp, analysis);
    return;
  }

  // linalg.generic operations may have multiple results and so for each result
  // we need to create a default lattice and emplace to the analysis state for
  // each result
  for (auto result : op.getResults()) {
    if (auto lattice = SparsityLattice::defaultFromValue(result)) {
      analysis.getState().try_emplace(result, lattice.value());
    }
  }

  // In the case where there are multiple results in the linalg.generic we will
  // not be moving forward with the forward pass, this may be implemented in
  // later versions
  if (op.getNumResults() != 1) {
    return;
  }

  // Here we check whether there already exists a lattice in the result
  // operation and attach that to the state, but it might not really make sense,
  // there may not be a usecase for this
  if (op.hasAttr("proteus.lattice")) {
    auto attr = op.getAttr("proteus.lattice");
    auto arrayAttr = llvm::cast<mlir::ArrayAttr>(attr);
    if (auto lattice = SparsityLattice::fromAttr(arrayAttr)) {
      if (auto *entry = analysis.getState(op.getResult(0))) {
        *entry = lattice.value();
      }
    }
  }

  // Now based on the mlir operation we dispatch the appropriate transfer
  // function
  mlir::TypeSwitch<mlir::Operation *>(&op)
      .Case<mlir::linalg::MatmulOp, mlir::linalg::AddOp, mlir::linalg::MatvecOp,
            mlir::linalg::VecmatOp, mlir::linalg::TransposeOp,
            mlir::linalg::BatchMatmulOp, mlir::linalg::FillOp,
            mlir::linalg::BroadcastOp, mlir::linalg::Conv2DOp,
            mlir::linalg::Conv2DNchwFchwOp, mlir::linalg::Conv2DNhwcHwcfOp,
            mlir::linalg::PoolingNchwMaxOp, mlir::linalg::PoolingNchwSumOp,
            mlir::linalg::DepthwiseConv2DNchwChwOp, mlir::tensor::PadOp,
            mlir::tensor::ConcatOp, mlir::tensor::ExpandShapeOp,
            mlir::tensor::ExtractSliceOp, mlir::tensor::CollapseShapeOp,
            mlir::linalg::GenericOp>(
          [&](auto typedOp) -> void { visitOp(typedOp, analysis); })
      .Case<mlir::linalg::AbsOp, mlir::linalg::CeilOp, mlir::linalg::FloorOp,
            mlir::linalg::NegFOp, mlir::linalg::DivOp,
            mlir::linalg::DivUnsignedOp, mlir::linalg::CopyOp,
            mlir::linalg::TanhOp, mlir::linalg::SquareOp, mlir::linalg::SqrtOp>(
          [&](auto) -> void { visitPassthroughOp(op, analysis); })
      .Default([](auto) {});
}

void proteus::ForwardPass::visitOp(mlir::linalg::MatmulOp &op,
                                   SparsityEngine &analysis) {
  auto *lhs = analysis.getState(op.getOperand(0));
  auto *rhs = analysis.getState(op.getOperand(1));
  auto *outs = analysis.getState(op.getDpsInits()[0]);
  auto *res = analysis.getState(op.getResult(0));

  // We want to keep any sparsity already in the result, this could happen
  // if an attribute is set on a matmul operation
  (*res)[0] &= (*lhs)[0];
  (*res)[1] &= (*rhs)[1];
  // Accumulator in the op
  (*res)[0] |= (*outs)[0];
  (*res)[1] |= (*outs)[1];
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
    (*res)[i] &= temp;
  }
}

void proteus::ForwardPass::visitOp(mlir::linalg::MatvecOp &op,
                                   SparsityEngine &analysis) {
  auto *lhs = analysis.getState(op->getOperand(0));
  auto *rhs = analysis.getState(op->getOperand(1));
  auto *outs = analysis.getState(op.getDpsInits()[0]);
  auto *res = analysis.getState(op->getResult(0));

  if ((*rhs)[0].none()) {
    (*res)[0].reset();
  } else {
    (*res)[0] &= (*lhs)[0];
  }

  // Accumulator in the op
  (*res)[0] |= (*outs)[0];
}

void proteus::ForwardPass::visitOp(mlir::linalg::VecmatOp &op,
                                   SparsityEngine &analysis) {
  auto *lhs = analysis.getState(op->getOperand(0));
  auto *rhs = analysis.getState(op->getOperand(1));
  auto *outs = analysis.getState(op.getDpsInits()[0]);
  auto *res = analysis.getState(op->getResult(0));

  if ((*lhs)[0].none()) {
    (*res)[0].reset();
  } else {
    (*res)[0] &= (*rhs)[1];
  }

  // Accumulator in the op
  (*res)[0] |= (*outs)[0];
}

void proteus::ForwardPass::visitOp(mlir::linalg::TransposeOp &op,
                                   SparsityEngine &analysis) {
  auto *input = analysis.getState(op.getOperand(0));
  auto *res = analysis.getState(op->getResult(0));

  llvm::ArrayRef<int64_t> perm = op.getPermutation();
  for (std::size_t i = 0; i < res->rank(); i++) {
    (*res)[i] &= (*input)[perm[i]];
  }
}

void proteus::ForwardPass::visitOp(mlir::linalg::BatchMatmulOp &op,
                                   SparsityEngine &analysis) {
  auto *lhs = analysis.getState(op->getOperand(0));
  auto *rhs = analysis.getState(op->getOperand(1));
  auto *outs = analysis.getState(op.getDpsInits()[0]);
  auto *res = analysis.getState(op->getResult(0));

  // Transfer sparsity for the batch dimension
  (*res)[0] &= (*lhs)[0];
  (*res)[0] &= (*rhs)[0];
  // Row slice sparsity is maintained from the lhs
  (*res)[1] &= (*lhs)[1];
  // Column slice sparsity in maintained from the rhs
  (*res)[2] &= (*rhs)[2];

  // Accumulator in the op
  (*res)[0] |= (*outs)[0];
  (*res)[1] |= (*outs)[1];
  (*res)[2] |= (*outs)[2];
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

  // True if the input has any dense bit in any dimension.
  bool hasDense =
      llvm::any_of(llvm::seq<std::size_t>(0, input->rank()),
                   [&](std::size_t d) { return (*input)[d].any(); });

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
      // If the input is fully sparse, every replica along this new
      // dimension is also all-zero, so we can clear it too.
      if (!hasDense) {
        (*res)[i].reset();
      }
      continue;
    }

    // If we do get to a dimension that existed previously in the input tensor
    // we propagate the sparsity to the current rank's bitvector, respecting
    // any preexisting lattice (e.g. an attribute set on the result).
    (*res)[i] &= (*input)[inputDim++];
  }
}

void proteus::ForwardPass::visitOp(mlir::tensor::PadOp &op,
                                   SparsityEngine &analysis) {
  auto *input = analysis.getState(op.getOperand(0));
  auto *res = analysis.getState(op.getResult());

  // Check all pads are static
  for (auto &pad : op.getMixedLowPad()) {
    if (!mlir::getConstantIntValue(pad)) {
      return;
    };
  }

  // Check all pads are static
  for (auto &pad : op.getMixedHighPad()) {
    if (!mlir::getConstantIntValue(pad)) {
      return;
    }
  }

  // Check constant padding
  auto cstOp =
      op.getConstantPaddingValue().getDefiningOp<mlir::arith::ConstantOp>();
  if (!cstOp) {
    return;
  }

  // Check whether padding is zero
  bool isPadZero = false;
  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(cstOp.getValue())) {
    isPadZero = floatAttr.getValue().isZero();
  } else if (auto intAttr =
                 mlir::dyn_cast<mlir::IntegerAttr>(cstOp.getValue())) {
    isPadZero = intAttr.getInt() == 0;
  }

  // If not we assume no propagation for simplicity
  // TODO: There should probably be a way where we could propagate something
  // here
  if (!isPadZero) {
    return;
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

  // Accumulate into a scratch lattice first, then meet it into res at the
  // end so that any preexisting lattice (e.g. an attribute set on the
  // result) is respected instead of being clobbered.
  SparsityLattice computed(res->shape());
  for (std::size_t i = 0; i < computed.rank(); ++i) {
    computed[i].reset();
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
    for (std::size_t j = 0; j < computed.rank(); j++) {
      if (j == concatDim) {
        // In the case of the concatenation dimension, we just concatenate
        // the bits of that dimension for all tensor operands
        for (std::size_t k = 0; k < (*operand)[j].size(); k++) {
          computed[j][k + offset] = (*operand)[j][k];
        }

        offset += (*operand)[j].size();
      } else {
        // For each operand we use the OR operation to make sure that only
        // in the case of all tensors being sparse on that dimension, the
        // sparsity is propagated
        computed[j] |= (*operand)[j];
      }
    }
  }

  for (std::size_t i = 0; i < res->rank(); ++i) {
    (*res)[i] &= computed[i];
  }
}

void proteus::ForwardPass::visitOp(mlir::linalg::Conv2DOp &op,
                                   SparsityEngine &analysis) {
  auto *input = analysis.getState(op.getOperand(0));
  auto *outs = analysis.getState(op.getDpsInits()[0]);
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

    // Accumulator in the op
    (*res)[i] |= (*outs)[i];
  }
}

void proteus::ForwardPass::visitOp(mlir::linalg::Conv2DNchwFchwOp &op,
                                   SparsityEngine &analysis) {
  auto *input = analysis.getState(op.getOperand(0));
  auto *filter = analysis.getState(op.getOperand(1));
  auto *outs = analysis.getState(op.getDpsInits()[0]);
  auto *res = analysis.getState(op.getResult(0));

  auto filterType =
      llvm::cast<mlir::RankedTensorType>(op.getOperand(1).getType());

  llvm::SmallVector<int64_t, 2> strides(op.getStrides().getValues<int64_t>());
  llvm::SmallVector<int64_t, 2> dilations(
      op.getDilations().getValues<int64_t>());

  // Accumulator in the op
  (*res)[1] &= (*filter)[0];
  (*res)[1] |= (*outs)[1];

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

      // Accumulator in the op
      (*res)[dim + 2] |= (*outs)[dim + 2];
    }
  }
}

void proteus::ForwardPass::visitOp(mlir::linalg::Conv2DNhwcHwcfOp &op,
                                   SparsityEngine &analysis) {
  auto *input = analysis.getState(op.getOperand(0));
  auto *filter = analysis.getState(op.getOperand(1));
  auto *outs = analysis.getState(op.getDpsInits()[0]);
  auto *res = analysis.getState(op.getResult(0));

  auto filterType =
      llvm::cast<mlir::RankedTensorType>(op.getOperand(1).getType());

  llvm::SmallVector<int64_t, 2> strides(op.getStrides().getValues<int64_t>());
  llvm::SmallVector<int64_t, 2> dilations(
      op.getDilations().getValues<int64_t>());

  (*res)[3] &= (*filter)[3];
  // Accumulator in the op
  (*res)[3] |= (*outs)[3];

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

      // Accumulator in the op
      (*res)[dim + 1] |= (*outs)[dim + 1];
    }
  }
}

void proteus::ForwardPass::visitOp(mlir::linalg::DepthwiseConv2DNchwChwOp &op,
                                   SparsityEngine &analysis) {
  auto *input = analysis.getState(op.getOperand(0));
  auto *filter = analysis.getState(op.getOperand(1));
  auto *outs = analysis.getState(op.getDpsInits()[0]);
  auto *res = analysis.getState(op.getResult(0));

  auto filterType =
      llvm::cast<mlir::RankedTensorType>(op.getOperand(1).getType());

  llvm::SmallVector<int64_t, 2> strides(op.getStrides().getValues<int64_t>());
  llvm::SmallVector<int64_t, 2> dilations(
      op.getDilations().getValues<int64_t>());

  // Channels convolve independently here, so a channel's conv contribution
  // is zero if either its input slice or its filter slice is entirely zero
  (*res)[1] &= (*input)[1];
  (*res)[1] &= (*filter)[0];
  // Accumulator in the op
  (*res)[1] |= (*outs)[1];

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

      // Accumulator in the op
      (*res)[dim + 2] |= (*outs)[dim + 2];
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

  // A dimension is truly dropped (rank-reduced) only when the result rank is
  // smaller than the source rank.  size==1 is necessary but not sufficient:
  // e.g. extracting [1, 58, 28, 28] from [1, 116, 28, 28] keeps all four dims.
  bool hasDimReduction = (res->rank() < offsets.size());

  uint64_t resDim = 0;
  for (uint64_t srcDim = 0; srcDim < offsets.size(); srcDim++) {
    auto size = mlir::getConstantIntValue(sizes[srcDim]);

    // Skip only when this dimension is actually being dropped from the result.
    if (hasDimReduction && size.has_value() && size.value() == 1) {
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

void proteus::ForwardPass::visitOp(mlir::tensor::CollapseShapeOp &op,
                                   SparsityEngine &analysis) {
  auto *input = analysis.getState(op.getSrc());
  auto *res = analysis.getState(op->getResult(0));

  auto inputType = mlir::cast<mlir::RankedTensorType>(op.getSrc().getType());
  auto resType = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // Reassociation is again an attribute attached to the tensor.collapse_shape
  // much like tensor.expand_shape that holds information about the way the
  // tensor will collapse into the resulting tensor
  auto reassociation = op.getReassociationIndices();

  for (auto [outDim, group] : llvm::enumerate(reassociation)) {
    int64_t outSize = resType.getDimSize(outDim);

    // Again we extract the row major single index coefficients like we did with
    // tensor.expand_shape
    llvm::SmallVector<int64_t> coefficients(group.size());
    coefficients.back() = 1;
    for (int p = static_cast<int>(group.size()) - 2; p >= 0; --p) {
      coefficients[p] =
          coefficients[p + 1] * inputType.getDimSize(group[p + 1]);
    }

    // Same as with what we did with tensor.expand_shape, we will create
    // an initial bitvector that we will explicitly construct from the sparsity
    // information of the corresponding fibers in the input tensor
    llvm::BitVector computed(outSize, false);
    for (int64_t k = 0; k < outSize; ++k) {
      bool allDense = true;

      for (auto [p, srcDim] : llvm::enumerate(group)) {
        int64_t srcFiber = (k / coefficients[p]) % inputType.getDimSize(srcDim);
        if (!(*input)[srcDim][srcFiber]) {
          allDense = false;
        }
      }

      if (allDense) {
        computed.set(k);
      }
    }

    (*res)[outDim] &= computed;
  }
}

void proteus::ForwardPass::visitOp(mlir::linalg::GenericOp &op,
                                   SparsityEngine &analysis) {

  if (mlir::succeeded(visitGenericReluOp(op, analysis))) {
    return;
  }

  if (mlir::succeeded(visitGenericClampOp(op, analysis))) {
    return;
  }

  if (mlir::succeeded(visitGenericAddFOp(op, analysis))) {
    return;
  }

  if (mlir::succeeded(visitGenericElementwiseZeroPreservingOp(op, analysis))) {
    return;
  }
}

mlir::LogicalResult
proteus::ForwardPass::visitGenericReluOp(mlir::linalg::GenericOp &op,
                                         SparsityEngine &analysis) {
  auto *body = op.getBody();
  auto &bodyOps = body->getOperations();
  auto it = bodyOps.begin();

  if (bodyOps.size() == 3 && mlir::isa<mlir::arith::CmpFOp>(*it) &&
      mlir::isa<mlir::arith::SelectOp>(*std::next(it))) {
    visitPassthroughOp(*op.getOperation(), analysis);
    return mlir::success();
  }

  return mlir::failure();
}

mlir::LogicalResult
proteus::ForwardPass::visitGenericClampOp(mlir::linalg::GenericOp &op,
                                          SparsityEngine &analysis) {
  auto *body = op.getBody();
  auto &bodyOps = body->getOperations();
  auto it = bodyOps.begin();

  auto allParallel = llvm::all_of(
      op.getIteratorTypesArray(), [](const mlir::utils::IteratorType t) {
        return t == mlir::utils::IteratorType::parallel;
      });

  if (!allParallel || op.getNumDpsInputs() != 3 || bodyOps.size() != 5) {
    return mlir::failure();
  }

  if (!mlir::isa<mlir::arith::CmpFOp>(*it) ||
      !mlir::isa<mlir::arith::SelectOp>(*std::next(it, 1)) ||
      !mlir::isa<mlir::arith::CmpFOp>(*std::next(it, 2)) ||
      !mlir::isa<mlir::arith::SelectOp>(*std::next(it, 3))) {
    return mlir::failure();
  }

  visitPassthroughOp(*op.getOperation(), analysis);
  return mlir::success();
}

mlir::LogicalResult
proteus::ForwardPass::visitGenericAddFOp(mlir::linalg::GenericOp &op,
                                         SparsityEngine &analysis) {
  auto *body = op.getBody();
  auto &bodyOps = body->getOperations();
  auto it = bodyOps.begin();

  if (bodyOps.size() != 2 || !mlir::isa<mlir::arith::AddFOp>(*it)) {
    return mlir::failure();
  }

  auto allParallel = llvm::all_of(
      op.getIteratorTypesArray(), [](const mlir::utils::IteratorType t) {
        return t == mlir::utils::IteratorType::parallel;
      });

  if (!allParallel || op.getNumDpsInputs() != 2) {
    return mlir::failure();
  }

  auto *lhs = analysis.getState(op.getOperand(0));
  auto *rhs = analysis.getState(op.getOperand(1));
  auto *res = analysis.getState(op.getResult(0));

  if (lhs->rank() != res->rank() || rhs->rank() != res->rank()) {
    return mlir::failure();
  }

  for (std::size_t i = 0; i < res->rank(); i++) {
    llvm::BitVector temp = (*lhs)[i];
    temp |= (*rhs)[i];
    (*res)[i] = temp;
  }

  return mlir::success();
}

mlir::LogicalResult
proteus::ForwardPass::visitGenericElementwiseZeroPreservingOp(
    mlir::linalg::GenericOp &op, SparsityEngine &analysis) {
  auto *body = op.getBody();
  auto &bodyOps = body->getOperations();

  auto allParallel = llvm::all_of(
      op.getIteratorTypesArray(), [](const mlir::utils::IteratorType t) {
        return t == mlir::utils::IteratorType::parallel;
      });

  auto allZeroPreserving =
      llvm::all_of(bodyOps, [](mlir::Operation &bodyOp) -> bool {
        return mlir::isa<
            mlir::linalg::YieldOp, mlir::arith::MulFOp, mlir::arith::MulIOp,
            mlir::arith::DivFOp, mlir::arith::DivSIOp, mlir::arith::DivUIOp,
            mlir::arith::NegFOp, mlir::arith::SIToFPOp, mlir::arith::UIToFPOp,
            mlir::arith::FPToSIOp, mlir::arith::FPToUIOp, mlir::arith::TruncFOp,
            mlir::arith::ExtFOp, mlir::arith::TruncIOp, mlir::arith::ExtSIOp,
            mlir::arith::ExtUIOp>(&bodyOp);
      });

  if (!allParallel || !allZeroPreserving) {
    return mlir::failure();
  }

  visitPassthroughOp(*op.getOperation(), analysis);
  return mlir::success();
}

void proteus::ForwardPass::visitPassthroughOp(mlir::Operation &op,
                                              SparsityEngine &analysis) {
  auto *operand = analysis.getState(op.getOperand(0));
  auto *res = analysis.getState(op.getOpResult(0));

  for (std::size_t i = 0; i < res->rank(); i++) {
    (*res)[i] = (*operand)[i];
  }
}
