"""
Setup script to compile CUDA metadata kernel

Usage:
    python setup_cuda_kernel.py install

Or for development:
    python setup_cuda_kernel.py build_ext --inplace
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='cuda_metadata_kernel',
    ext_modules=[
        CUDAExtension(
            name='cuda_metadata_kernel',
            sources=[
                os.path.join(current_dir, 'cuda_metadata_kernel.cu'),
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-lineinfo',
                    # Generate code for multiple architectures
                    '-gencode', 'arch=compute_70,code=sm_70',  # V100
                    '-gencode', 'arch=compute_75,code=sm_75',  # T4, RTX 20xx
                    '-gencode', 'arch=compute_80,code=sm_80',  # A100
                    '-gencode', 'arch=compute_86,code=sm_86',  # RTX 30xx
                    '-gencode', 'arch=compute_89,code=sm_89',  # RTX 40xx
                    '-gencode', 'arch=compute_90,code=sm_90',  # H100
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
    ],
)
