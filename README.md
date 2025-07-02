# Fork of NXP's [TfLite-ethosu-delegate](https://github.com/nxp-imx/tflite-ethosu-delegate-imx) with added support for QNX OS

TfLite-ethosu-delegate is a LiteRT (TFLite) delegate.
This fork extends the functionality to also support QNX OS and provides CMakePresets.json for easier CMake configuration process.

# Use tflite-ethosu-delegate for Linux

## Prepare source code
```sh
mkdir wksp && cd wksp
git clone https://github.com/nxpmicro/tflite-ethosu-delegate.git
# tensorflow is optional, it will be downloaded automatically if not present
git clone https://github.com/tensorflow/tensorflow.git
```
# Build from source with CMake (for Linux)

```sh
# set the toolchain env
source /PATH_TO_TOOLCHAIN/environment-setup-cortexa53-crypto-poky-linux

# build the delegate
cd tflite-ethosu-delegate
cmake --preset linux
cmake --build --preset linux

cd build_linux

# benchmark_model
make benchmark_model -j12
# label_image
make label_image -j12
```

# Build from source with CMake (for QNX)

```sh
cd tflite-ethosu-delegate
cmake --preset qnx
cmake --build --preset qnx

cd build_qnx

# benchmark_model
make benchmark_model -j12
# label_image
make label_image -j12
```

---

If you would like to build using local version of tensorflow, you can use `FETCHCONTENT_SOURCE_DIR_TENSORFLOW` cmake variable. Point this variable to your tensorflow tree. For additional details on this variable please see the [official cmake documentation](https://cmake.org/cmake/help/latest/module/FetchContent.html#command:fetchcontent_populate)

``` sh
cmake -DFETCHCONTENT_SOURCE_DIR_TENSORFLOW=/my/copy/of/tensorflow \
    -DOTHER_CMAKE_DEFINES...\
    ..
```
After cmake execution completes, build and run as usual.

## Enable external delegate support in benchmark_model/label_image

If tensorflow source code downloaded by cmake, you can find it in <build_output_dir>/_deps/tensorflow-src

## Run
```sh
./benchmark_model --external_delegate_path=<patch_to_libethosu_delegate.so> --graph=<tflite_model.tflite>
# If you would like to enable PMU counter
./benchmark_model --external_delegate_path=<patch_to_libethosu_delegate.so> \
                  --external_delegate_options='enable_cycle_counter:true;pmu_event0:3;pmu_event1:4;pmu_event2:5;pmu_event3:6' \
                  --graph=<tflite_model.tflite>
```
