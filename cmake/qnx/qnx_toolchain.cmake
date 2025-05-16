# Platform defines.
set(CMAKE_CROSSCOMPILING 1)
set(CMAKE_SYSTEM_PROCESSOR aarch64 CACHE STRING "System Processor")
set(DE_CPU "DE_CPU_ARM_64")
set(DE_OS "DE_OS_QNX")
set(DE_PTR_SIZE 8)
set(QNXNTO TRUE)

set(arch "aarch64le")

set(DE_COMPILER "DE_COMPILER_GCC")

# qcc did not work for some reason when compiling ARM assembly from XNNPACK
# compiling using this instead worked
set(CMAKE_ASM_COMPILER "ntoaarch64-gcc")

set(CMAKE_C_COMPILER "qcc")
set(CMAKE_C_COMPILER_TARGET "gcc_ntoaarch64le")

set(CMAKE_CXX_COMPILER "q++")
set(CMAKE_CXX_COMPILER_TARGET "gcc_ntoaarch64le_cxx")

set(CMAKE_LIBRARY_PATH $ENV{QNX_TARGET}/${arch}/usr/lib $ENV{QNX_TARGET}/${arch}/lib)

set(QNX_INCLUDE_PATH $ENV{QNX_TARGET}/usr/include)
set(ZLIB_INCLUDE_PATH $ENV{QNX_TARGET}/usr/include)
set(PNG_INCLUDE_PATH $ENV{QNX_TARGET}/usr/include)

set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE
    "-Wl,--whole-archive"
    "<LINK_ITEM>"
    "-Wl,--no-whole-archive"
)
set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE_SUPPORTED true)

set(CMAKE_FIND_ROOT_PATH 
    $ENV{VIVANTE_SDK_LIB}
    $ENV{QNX_TARGET}/${arch}/usr/lib
    )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY FIRST)
set(CMAKE_SYSROOT $ENV{QNX_TARGET})
set(CMAKE_CXX_FLAGS "-Wall -O2 -std=c++17")
set(CMAKE_C_FLAGS "-Wall -O2")

# defines _SC_NPROCESSORS_CONF for pthreadpool
add_definitions(-D_QNX_SOURCE)

# defines for farmhash, from what I saw QNX has bultin bswap and is little endian (for some reason farmhash does not define this)
add_definitions(-DHAVE_BUILTIN_BSWAP)
add_definitions(-DFARMHASH_LITTLE_ENDIAN)
