set(GLFW_DIR "../deps/glfw")

include_directories(
	"../deps"
	"../kdRay"
    "../deps/glew/include"
    "nativefiledialog/include"
	${GLFW_DIR}/include
)

add_definitions(
    -DGLEW_STATIC)

set(HEADER_FILES
)

find_library(libGLFW glfw3 ${GLFW_DIR}/lib)
if(EXISTS ${libGLFW})
    message("find glfw: ${libGLFW}")
else()
    message(FATAL_ERROR "glfw not find: ${libGLFW}")
endif()

find_library(libGL opengl32)
if(EXISTS ${libGL})
    message("find opengl: ${libGL}")
else()
    message(FATAL_ERROR "opengl not find: ${libGL}")
endif()

set(HEADER_FILES
)

aux_source_directory("." SRC)
aux_source_directory("./imgui" IMGUI_SRC)
set(GLEW_SRC "../deps/glew/src/glew.c")
set(NFD_SRC "nativefiledialog/nfd_common.c" "nativefiledialog/nfd_win.cpp")

add_executable(kdRayViewer ${HEADER_FILES} ${SRC} ${GLEW_SRC} ${IMGUI_SRC} ${NFD_SRC})
target_link_libraries(kdRayViewer libKDRay ${libGLFW} ${libGL})
