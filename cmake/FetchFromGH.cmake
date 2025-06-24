cmake_minimum_required(VERSION 3.24)

function(download_github_release)
    set(options "")
    set(oneValueArgs NAME REPO TAG FORCE_DL)
    set(multiValueArgs "")
    
    cmake_parse_arguments(DGR "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # Validate required arguments
    if(NOT DGR_NAME)
        message(FATAL_ERROR "NAME is required")
    endif()
    
    if(NOT DGR_REPO)
        message(FATAL_ERROR "REPO is required")
    endif()
    
    if(NOT DGR_TAG)
        message(FATAL_ERROR "TAG is required")
    endif()

    if(NOT DGR_FORCE_DL)
        set(DGR_FORCE_DL FALSE)
    endif()
    
    # Determine platform-specific asset pattern
    if(WIN32)
        set(ASSET_PATTERN "${DGR_NAME}-windows.zip")
    elseif(UNIX AND NOT APPLE)
        set(ASSET_PATTERN "${DGR_NAME}-linux.tar.gz")
    elseif(APPLE)
        set(ASSET_PATTERN "${DGR_NAME}-macos.tar.gz")
    else()
        message(FATAL_ERROR "Unsupported platform")
    endif()
    
    set(RELEASE_URL "https://github.com/${DGR_REPO}/releases/download/${DGR_TAG}/${ASSET_PATTERN}")
    
    include(FetchContent)
    string(TOLOWER ${DGR_NAME} LOWER_DGR_NAME)
    # Check if already populated
    FetchContent_GetProperties(${DGR_NAME})

    if(NOT ${DGR_NAME}_DIR OR ${DGR_FORCE_DL})
        message(STATUS "[${DGR_NAME}] Downloading and extracting from ${RELEASE_URL}")
        
        FetchContent_Declare(
            ${DGR_NAME}
            URL ${RELEASE_URL}
            DOWNLOAD_EXTRACT_TIMESTAMP OFF
        )
        
        FetchContent_MakeAvailable(${DGR_NAME})
    else()
        message(STATUS "Using ${DGR_NAME} found here: ${${DGR_NAME}_DIR}")
    endif()
    
    set(${DGR_NAME}_CMAKE_DIR ${${LOWER_DGR_NAME}_SOURCE_DIR}/lib/cmake)
    list(APPEND CMAKE_PREFIX_PATH  "${${DGR_NAME}_CMAKE_DIR}")

    find_package(${DGR_NAME} REQUIRED)
endfunction()