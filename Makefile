.PHONY: all config build config-release build-release set-style format test tidy watch clean docs \
        dataset-convert dataset-clean coverage-check \
        lit-coverage lit-coverage-check lit-coverage-clean submodules experiments models-fetch

NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.logicalcpu)
COVERAGE_MIN ?= 85
SPLAT_WEIGHTS ?=
LATTICE_FILTER ?=
COVERAGE_SOURCES := $(shell find lib/Analysis lib/Runtime -name "*.cpp")
RUNTIME_LIB_GLOB = $$(find build-cov/lib -name "libProteusProbeRuntime.*")

submodules: external/mlir-probe/.git

external/mlir-probe/.git:
	git submodule update --init --recursive -- external/mlir-probe

external/bennu/.git:
	git submodule update --init -- external/bennu

models-fetch: external/bennu/.git
	@command -v git-lfs >/dev/null 2>&1 || { echo "Error: git-lfs is required to fetch models. Install it with: brew install git-lfs" >&2; exit 1; }
	printf '*.onnx filter=lfs diff=lfs merge=lfs -text\n' > external/bennu/.gitattributes
	git -C external/bennu lfs pull
	mkdir -p models
	cp -f external/bennu/models/*.onnx models/

config: submodules
	cmake -S . -B build -G Ninja

build/build.ninja: CMakeLists.txt lib/CMakeLists.txt tools/proteus-opt/CMakeLists.txt tests/CMakeLists.txt tests/unit/CMakeLists.txt external/mlir-probe/.git
	cmake -S . -B build -G Ninja

build: build/build.ninja
	cmake --build build --parallel $(NPROC)

config-release: submodules
	cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release

build-release/build.ninja: CMakeLists.txt lib/CMakeLists.txt tools/proteus-opt/CMakeLists.txt tests/CMakeLists.txt tests/unit/CMakeLists.txt external/mlir-probe/.git
	cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release

build-release: build-release/build.ninja
	cmake --build build-release --parallel $(NPROC)

set-style:
	@clang-format --style=LLVM --dump-config > .clang-format

format:
	@find lib include tests \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" \) \
		| xargs clang-format -i

format-check:
	@find lib include tests \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" \) \
		| xargs clang-format --dry-run --Werror

test: build
	ctest --verbose --test-dir build --output-on-failure
	lit tests

tidy:
	find lib tools -name "*.cpp" | xargs clang-tidy -p build

watch:
	watchman-make -p 'lib/**/*.cpp' 'include/**/*.h' 'tools/**/*.cpp' -t tidy

clean:
	rm -rf build

docs: build
	doxygen

dataset-convert: models-fetch
	mkdir -p mlir_out mlir_out_zerobias
	docker compose run --build --rm -e SPLAT_WEIGHTS=$(SPLAT_WEIGHTS) convert

dataset-clean:
	rm -rf mlir_out mlir_out_zerobias

experiments:
	docker compose run --build --rm -e LATTICE_FILTER=$(LATTICE_FILTER) run

coverage-check:
	python3 scripts/check_forward_pass_coverage.py

lit-coverage:
	cmake -S . -B build-cov -G Ninja \
		-DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping -fprofile-continuous" \
		-DCMAKE_C_FLAGS="-fprofile-instr-generate -fcoverage-mapping -fprofile-continuous" \
		-DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-generate -fprofile-continuous" \
		-DCMAKE_SHARED_LINKER_FLAGS="-fprofile-instr-generate -fprofile-continuous"
	cmake --build build-cov --parallel $(NPROC)
	rm -rf build-cov/profiles && mkdir -p build-cov/profiles
	PROTEUS_BUILD_DIR=build-cov \
		LLVM_PROFILE_FILE="$(CURDIR)/build-cov/profiles/proteus-%p.profraw" \
		lit -j1 tests
	llvm-profdata merge -sparse build-cov/profiles/*.profraw -o build-cov/proteus.profdata
	llvm-cov report build-cov/bin/proteus-opt \
		--object $(RUNTIME_LIB_GLOB) \
		-instr-profile=build-cov/proteus.profdata \
		$(COVERAGE_SOURCES)
	llvm-cov show build-cov/bin/proteus-opt \
		--object $(RUNTIME_LIB_GLOB) \
		-instr-profile=build-cov/proteus.profdata \
		-format=html -output-dir=build-cov/coverage-html \
		$(COVERAGE_SOURCES)
	@echo "HTML report: build-cov/coverage-html/index.html"

lit-coverage-check: lit-coverage
	llvm-cov export -summary-only build-cov/bin/proteus-opt \
		--object $(RUNTIME_LIB_GLOB) \
		-instr-profile=build-cov/proteus.profdata \
		$(COVERAGE_SOURCES) \
		| python3 scripts/check_coverage_threshold.py $(COVERAGE_MIN)

lit-coverage-clean:
	rm -rf build-cov
