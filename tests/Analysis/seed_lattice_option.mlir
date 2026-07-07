// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=seed seed-lattice='[{size = 2 : i64, words = array<i64: 1>}, {size = 2 : i64, words = array<i64: 3>}]'" %s | FileCheck %s
// RUN: not proteus-opt --spa-analysis="seed-lattice='not an attr'" %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR

// CHECK-LABEL: func.func @seed_lattice_test
// CHECK-SAME: %arg0: tensor<2x2xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 1>}, {size = 2 : i64, words = array<i64: 3>}]}

// CHECK-ERR: seed-lattice must parse to an ArrayAttr

func.func @seed_lattice_test(
    %a : tensor<2x2xf32>
) -> tensor<2x2xf32> {
  return %a : tensor<2x2xf32>
}
