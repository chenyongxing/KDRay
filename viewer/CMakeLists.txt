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

find_library(libGL opengl32 REQUIRED)
find_library(libGLFW glfw3 ${GLFW_DIR}/lib)

aux_source_directory("." SRC)
aux_source_directory("./imgui" IMGUI_SRC)
set(GLEW_SRC "../deps/glew/src/glew.c")
set(NFD_SRC "nativefiledialog/nfd_common.c" "nativefiledialog/nfd_win.cpp")

add_executable(kdRayViewer ${SRC} ${GLEW_SRC} ${IMGUI_SRC} ${NFD_SRC})
target_link_libraries(kdRayViewer libKDRay ${libGLFW} ${libGL})
