cmake -S . -B build   -DPython_EXECUTABLE="$(which python)" -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.4

cmake --build build -j