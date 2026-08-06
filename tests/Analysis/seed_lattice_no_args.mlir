// RUN: not proteus-opt --spa-analysis="seed-lattice='[{size = 2 : i64, words = array<i64: 1>}]'" %s 2>&1 | FileCheck %s

// CHECK: spa-analysis: seed-lattice given but function has no arguments

func.func @seed_lattice_no_args_test() -> () {
  return
}
