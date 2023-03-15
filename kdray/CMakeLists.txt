find_package(CUDA 11 REQUIRED)
message("find cuda include dir: " ${CUDA_INCLUDE_DIRS})

set(OPTIX7_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.6.0")
if(EXISTS ${OPTIX7_DIR}/include/optix.h)
    message("find optix7 dir: ${OPTIX7_DIR}")
else()
    message(FATAL_ERROR "optix7 dir not find: ${OPTIX7_DIR}")
endif()

#编译cu到ptx，然后bin2c把ptx写入c文件
find_program(BIN2C bin2c)
message("find bin2c: ${BIN2C}")
macro(cuda_compile_and_embed output_var cuda_file)
  set(c_var_name ${output_var})
  cuda_compile_ptx(ptx_files ${cuda_file} OPTIONS -arch compute_60 -use_fast_math -lineinfo -allow-unsupported-compiler -rdc true -D__x86_64 -DOPTIX_OPTIONAL_FEATURE_OPTIX7)
  list(GET ptx_files 0 ptx_file)
  set(embedded_file ${ptx_file}_embedded.c)
  message("compile ptx and embed ${cuda_file} to \"const char ${c_var_name}[];\"")
  add_custom_command(
    OUTPUT ${embedded_file}
    COMMAND ${BIN2C} -c --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
    DEPENDS ${ptx_file}
    )
  set(${output_var} ${embedded_file})
endmacro()

include_directories(
	"../deps"
    ${CUDA_INCLUDE_DIRS}
    ${OPTIX7_DIR}/include
)

set(HEADER_FILES
    logger.h
    kdray.h
    scene.h
    optix_context.h
)

aux_source_directory("." SOURCE_FILES)

#optix-ptx
cuda_compile_and_embed(cuRayTracer cuda/main.cu)
#cu
cuda_add_library(cuToneMap cuda/tone_map.cu)

add_library(libKDRay SHARED ${HEADER_FILES} ${SOURCE_FILES} ${cuRayTracer})
target_link_libraries(libKDRay cuToneMap ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
