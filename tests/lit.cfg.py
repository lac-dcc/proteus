import lit.formats
import os

config.name = "SPA"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir']

config.test_source_root = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(config.test_source_root)
build_bin = os.path.join(project_root, 'build', 'bin')

llvm_prefix = os.environ.get('LLVM_PREFIX', '')
llvm_bin = os.path.join(llvm_prefix, 'bin') if llvm_prefix else ''

path_dirs = [build_bin]
if llvm_bin:
    path_dirs.append(llvm_bin)
path_dirs.append(os.environ.get('PATH', ''))

config.environment['PATH'] = os.pathsep.join(path_dirs)
