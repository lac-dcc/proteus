.PHONY: all config build set-style format test tidy watch clean docs

config:
	cmake -S . -B build

build: config
	cmake --build build

set-style:
	@clang-format --style=LLVM --dump-config > .clang-format

format: set-style
	@find lib include tests \( -name "*.cpp" -o -name "*.h" \) \
		| xargs clang-format -i

test: build
	lit tests

tidy:
	cmake --build build --target clang-tidy

watch:
	watchman-make -p 'lib/**/*.cpp' 'include/**/*.h' 'tools/**/*.cpp' -t tidy

clean:
	rm -rf build

docs: build
	doxygen
