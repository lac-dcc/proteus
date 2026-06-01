.PHONY: all config build set-style format test tidy watch clean docs

NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.logicalcpu)

config:
	cmake -S . -B build -G Ninja

build/build.ninja: CMakeLists.txt lib/CMakeLists.txt tools/proteus-opt/CMakeLists.txt tests/CMakeLists.txt tests/unit/CMakeLists.txt
	cmake -S . -B build -G Ninja

build: build/build.ninja
	cmake --build build --parallel $(NPROC)

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
