rm -r ./build
mkdir -p build
cp ./*.bin ./build/
cd build
cmake ..
cmake --build .