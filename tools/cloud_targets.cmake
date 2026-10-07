# The persistent benchmark needs the public CPU pool and registry APIs in
# addition to llama; these are separate DSOs in the pinned shared build.
find_library(CPU_DECODE_GGML_CPU_LIBRARY NAMES ggml-cpu
  PATHS "${CPU_DECODE_LLAMA_ROOT}/build/bin" NO_DEFAULT_PATH)
find_library(CPU_DECODE_GGML_BASE_LIBRARY NAMES ggml-base
  PATHS "${CPU_DECODE_LLAMA_ROOT}/build/bin" NO_DEFAULT_PATH)
if(NOT CPU_DECODE_GGML_CPU_LIBRARY OR NOT CPU_DECODE_GGML_BASE_LIBRARY)
  message(FATAL_ERROR "cloud-bench requires the pinned prebuilt ggml-cpu and ggml-base libraries")
endif()
add_executable(cloud-bench tools/cloud_bench.cpp)
target_include_directories(cloud-bench SYSTEM PRIVATE
  "${CPU_DECODE_LLAMA_INCLUDE_DIR}" "${CPU_DECODE_GGML_INCLUDE_DIR}")
target_link_libraries(cloud-bench PRIVATE decode-core cpu-decode-llama
  "${CPU_DECODE_GGML_CPU_LIBRARY}" "${CPU_DECODE_GGML_BASE_LIBRARY}")
target_compile_options(cloud-bench PRIVATE -O3 -Wall -Wextra -Wpedantic)
set_target_properties(cloud-bench PROPERTIES
  BUILD_RPATH "${CPU_DECODE_LLAMA_LIBRARY_DIR}"
  INSTALL_RPATH "${CPU_DECODE_LLAMA_LIBRARY_DIR}")
