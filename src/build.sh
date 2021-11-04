rm -r ./build
mkdir -p build
cp ./*.bin ./build/
cd build
cmake ..
cmake --build .
# salloc -N1 -n1 -t10
# cd /path/to/build/
# srun ./cuda