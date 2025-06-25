cmake_minimum_required (VERSION 3.21)

project("RLGymPPO_CPP" VERSION 1.0.2 LANGUAGES CXX)

include(FetchContent)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(Torch_DIR "../RandomProjects/deps/libtorch/share/cmake/Torch")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(FetchFromGH)

DOWNLOAD_GITHUB_RELEASE(
    NAME RLGymSim_CPP
    REPO MrCrystAll/RLGymSim_CPP
    TAG v1.0.6
)

find_package(RLGymSim_CPP REQUIRED)

include_directories("${PROJECT_SOURCE_DIR}/src/")

# Make sure CMake finds libtorch if its in this directory
if (EXISTS "${PROJECT_SOURCE_DIR}/libtorch/")
	message("Using local libtorch folder...")
	list(APPEND CMAKE_PREFIX_PATH "libtorch")
	# Make ultra-sure we can find libtorch if its local
	set(CMAKE_PREFIX_PATH "libtorch/share/cmake/Torch")
endif()

# Add libtorch (https://pytorch.org/cppdocs/installing.html#minimal-example)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add all headers and code files
file(GLOB_RECURSE FILES_SRC "src/*.cpp")

add_library(RLGymPPO_CPP SHARED ${FILES_SRC})
target_compile_definitions(RLGymPPO_CPP PRIVATE -DWITHIN_RLGPC)


target_include_directories(${PROJECT_NAME}
		PUBLIC
		$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
		$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libsrc>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
		$<INSTALL_INTERFACE:include>
        PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/private>
)

# Include libtorch
target_link_libraries(RLGymPPO_CPP PRIVATE "${TORCH_LIBRARIES}" ${RLGymSim_CPP_LIBRARIES} ${RocketSim_LIBRARIES})

if (TORCH_CUDA_LIBRARIES)
	message("Enabling CUDA support...")
	target_compile_definitions(RLGymPPO_CPP PRIVATE -DRG_CUDA_SUPPORT)
endif()

# Set C++ version to 20
set_target_properties(RLGymPPO_CPP PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RLGymPPO_CPP PROPERTIES CXX_STANDARD 20)

# Include JSON
#target_include_directories(RLGymPPO_CPP PRIVATE "${PROJECT_SOURCE_DIR}/libsrc/json")

# Include python
find_package(Python3 3.7...<3.11 COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})
target_link_libraries(RLGymPPO_CPP PUBLIC ${Python3_LIBRARIES})
message("Found Python:")
message("PYTHON_LIBRARIES: ${Python3_LIBRARIES}")
message("PYTHON_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message("Python_RUNTIME_LIBRARY_DIRS: ${Python3_RUNTIME_LIBRARY_DIRS}")
message("Python_EXECUTABLE: ${Python3_EXECUTABLE}")

set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

add_definitions(-DPY_EXEC_PATH="${Python3_EXECUTABLE}") # Give C++ access to the executable path

# Pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# Include pybind11
target_link_libraries(RLGymPPO_CPP PUBLIC pybind11::embed)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/RLGymPPO_CPP/config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/RLGymPPO_CPP/config.h"
    @ONLY
)

# MSVC fails to find python DLLs even through they are in my path. Good job MSVC. Well done.
# This copies the the python DLLs to the output directory
if (MSVC)
    file(GLOB PYTHON_DLLS "${Python3_RUNTIME_LIBRARY_DIRS}/*.dll")
	message("Adding Python DLLS: ${PYTHON_DLLS}")
    add_custom_command(TARGET RLGymPPO_CPP
                 POST_BUILD
                 COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 ${PYTHON_DLLS}
                 $<TARGET_FILE_DIR:RLGymPPO_CPP>)
endif (MSVC)

# Make our python files copy over to our build dir
configure_file("./python_scripts/metric_receiver.py" "./python_scripts/metric_receiver.py" COPY)
configure_file("./python_scripts/render_receiver.py" "./python_scripts/render_receiver.py" COPY)

# MSVC sometimes won't link to the libtorch DLLs unless you do this
# This is also from https://pytorch.org/cppdocs/installing.html#minimal-example
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
	message(STATUS "Adding TORCH_DLLS: ${TORCH_DLLS}")
    add_custom_command(TARGET RLGymPPO_CPP
                 POST_BUILD
                 COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 ${TORCH_DLLS}
                 $<TARGET_FILE_DIR:RLGymPPO_CPP>)

    if(MSVC)
        file(GLOB RLGymSim_CPP_DLLS "${RLGymSim_CPP_BIN_DIR}/*.dll")
    elseif(UNIX)
        file(GLOB RLGymSim_CPP_DLLS "${RLGymSim_CPP_LIB_DIR}/*.so")
    endif()

    message(STATUS "Adding RLGymSim_CPP_DLLS: ${RLGymSim_CPP_DLLS}")
    add_custom_command(TARGET RLGymPPO_CPP
                 POST_BUILD
                 COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 ${RLGymSim_CPP_DLLS}
                 $<TARGET_FILE_DIR:RLGymPPO_CPP>)
endif (MSVC)