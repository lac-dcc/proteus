import glob
import lit.formats
import os

config.name = "SPA"
config.test_format = lit.formats.ShTest(False)
config.suffixes = ['.mlir']

config.test_source_root = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(config.test_source_root)
build_dir = os.environ.get('PROTEUS_BUILD_DIR', 'build')
build_bin = os.path.join(project_root, build_dir, 'bin')

llvm_prefix = os.environ.get('LLVM_PREFIX', '')
llvm_bin = os.path.join(llvm_prefix, 'bin') if llvm_prefix else ''

path_dirs = [build_bin]
if llvm_bin:
    path_dirs.append(llvm_bin)
path_dirs.append(os.environ.get('PATH', ''))

config.environment['PATH'] = os.pathsep.join(path_dirs)

# Shared library extension (.so/.dylib/...) varies by platform, so glob for
# whatever CMake actually produced instead of hard-coding a suffix.
runtime_libs = glob.glob(
    os.path.join(project_root, build_dir, 'lib', 'libProteusProbeRuntime.*'))
if runtime_libs:
    config.substitutions.append(('%proteus_runtime_lib', runtime_libs[0]))

if llvm_prefix:
    for name, subst in (('libmlir_runner_utils', '%mlir_runner_utils'),
                        ('libmlir_c_runner_utils', '%mlir_c_runner_utils')):
        libs = glob.glob(os.path.join(llvm_prefix, 'lib', name + '.*'))
        if libs:
            config.substitutions.append((subst, libs[0]))
