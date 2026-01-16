#!/bin/bash
#
# Script to compile CUDA metadata kernel
#

set -e  # Exit on error

echo "=========================================="
echo "Building CUDA Metadata Kernel"
echo "=========================================="

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "Step 1: Checking CUDA availability..."
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')"

echo ""
echo "Step 2: Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info cuda_metadata_kernel*.so

echo ""
echo "Step 3: Compiling CUDA kernel..."
python3 setup_cuda_kernel.py build_ext --inplace

echo ""
echo "Step 4: Verifying compilation..."
if [ -f "cuda_metadata_kernel*.so" ] || [ -f "cuda_metadata_kernel.*.so" ]; then
    echo "✅ Compilation successful!"
    ls -lh cuda_metadata_kernel*.so
else
    echo "❌ Compilation failed - .so file not found"
    exit 1
fi

echo ""
echo "Step 5: Running self-test..."
python3 cuda_metadata_wrapper.py

echo ""
echo "=========================================="
echo "✅ Build Complete!"
echo "=========================================="
echo ""
echo "The CUDA kernel is now ready to use."
echo "To integrate with nsa_backend.py, the kernel will be"
echo "automatically detected and used if available."
echo ""
