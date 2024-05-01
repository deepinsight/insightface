mkdir -p build/local
# shellcheck disable=SC2164
cd build/local
# export cross_compile_toolchain=/home/s4129/software/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf
cmake -DCMAKE_BUILD_TYPE=Release \
  -DTHIRD_PARTY_DIR=3rdparty \
  -DBUILD_WITH_SAMPLE=ON \
  -DBUILD_WITH_TEST=OFF \
  -DENABLE_BENCHMARK=OFF \
  -DENABLE_USE_LFW_DATA=OFF \
  -DENABLE_TEST_EVALUATION=OFF \
  -DBUILD_SHARED_LIBS=ON ../../

make -j4
make install