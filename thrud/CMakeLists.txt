set(THRUD "Thrud")
set(LIB_DIR "lib")
get_filename_component(INCLUDE_DIR "include" ABSOLUTE)

# Find llvm.
set(LLVM_DIR "/data/build/llvm/share/llvm/cmake/")
find_package(LLVM REQUIRED)
if (NOT LLVM_FOUND)
  message(FATAL_ERROR "LLVM NOT FOUND")
endif (NOT LLVM_FOUND)

# Include dirs.
include_directories(${INCLUDE_DIR})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})

message(STATUS "Include dir: ${INCLUDE_DIR}")
message(STATUS "LLVM include dirs: ${LLVM_INCLUDE_DIRS}")
message(STATUS "LLVM library dirs: ${LLVM_LIBRARY_DIRS}")

# Link dirs.
link_directories(${LLVM_LIBRARY_DIRS})

# Definitions.
add_definitions("-D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS")

# Subdir.
add_subdirectory(${LIB_DIR})
