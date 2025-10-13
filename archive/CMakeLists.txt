cmake_minimum_required(VERSION 3.20)
project(controller LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optional: treat warnings as errors in development
add_compile_options(-Wall -Wextra -Wpedantic)

# === Part 1: Core library ===
add_library(controller_core
    loading_controller.cpp
    storage_selector.cpp
)

target_include_directories(controller_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# === Part 2: Main binder executable ===
add_executable(bond
    bond.cpp
)
target_link_libraries(bond PRIVATE controller_core)

# === Part 3: Tests ===
add_executable(test_loading_controller
    test_loading_controller.cpp
    test_utils.h
)
target_link_libraries(test_loading_controller PRIVATE controller_core)

add_executable(test_storage_selector
    test_storage_selector.cpp
    test_utils.h
)
target_link_libraries(test_storage_selector PRIVATE controller_core)

# Optional: convenience target to run all tests
add_custom_target(run_tests
    DEPENDS test_loading_controller test_storage_selector
    COMMAND test_loading_controller
    COMMAND test_storage_selector
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
