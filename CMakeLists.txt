cmake_minimum_required(VERSION 3.10)

project(zero VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

if(WIN32)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_GLFW_WIN32 -D_MBCS -DNDEBUG -DNOMINMAX -DWIN32_LEAN_AND_MEAN)
  if (EXISTS lib/glfw/src)
    target_compile_definitions(zero PUBLIC GLFW_AVAILABLE=1)
    set(GLFW_SOURCES
        lib/glfw/src/context.cpp
        lib/glfw/src/egl_context.cpp
        lib/glfw/src/init.cpp
        lib/glfw/src/input.cpp
        lib/glfw/src/monitor.cpp
        lib/glfw/src/osmesa_context.cpp
        lib/glfw/src/vulkan.cpp
        lib/glfw/src/wgl_context.cpp
        lib/glfw/src/win32_init.cpp
        lib/glfw/src/win32_joystick.cpp
        lib/glfw/src/win32_monitor.cpp
        lib/glfw/src/win32_thread.cpp
        lib/glfw/src/win32_time.cpp
        lib/glfw/src/win32_window.cpp
        lib/glfw/src/window.cpp)
  endif()
elseif(UNIX)
  add_definitions(-DNDEBUG)
endif()

include(GNUInstallDirs)

file(GLOB_RECURSE SOURCES zero/*.cpp)
list(APPEND SOURCES lib/glad/src/glad.cpp)
list(APPEND SOURCES ${GLFW_SOURCES})

add_executable(zero ${SOURCES})
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # Use parallel compilation
  target_compile_options(zero PRIVATE "/MP")
endif()

#target_compile_options(zero PRIVATE "-ftime-trace")

target_include_directories(zero PRIVATE
                           .
                           lib
                           lib/glad/include
                           lib/glfw/include)

if(WIN32)
  target_link_libraries(zero ws2_32)
else()
  find_package(glfw3 3.3 QUIET)
  if(glfw3_FOUND)
    message(STATUS "Using GLFW3")
    target_link_libraries(zero glfw dl -pthread)
  target_compile_definitions(zero PUBLIC GLFW_AVAILABLE=1)
  else()
    message(WARNING "GLFW3 not found. Render window disabled.")
    target_link_libraries(zero dl -pthread)
  endif()
endif()

set(CPACK_PACKAGE_NAME "zero")
set(CPACK_PACKAGE_VENDOR "plushmonkey")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "zero - Continuum bot")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "zero")

set(CMAKE_INSTALL_PREFIX ".")

install(TARGETS zero
        CONFIGURATIONS Debug
        RUNTIME DESTINATION Debug)

install(TARGETS zero
        CONFIGURATIONS Release
        RUNTIME DESTINATION Release)

# The headless duel simulator (bot-ai ticket 41): zero's sources but its main, plus sim/.
set(SIM_SOURCES ${SOURCES})
list(FILTER SIM_SOURCES EXCLUDE REGEX ".*/zero/main\\.cpp$")
# Of the zones, only Chaos, whose bots the simulator can run (ticket 45): the other zones' global controllers listen on
# the event bus and expect a bot. ZERO_SIM leaves out Chaos's own global controller; the simulator makes one a bot.
list(FILTER SIM_SOURCES EXCLUDE REGEX ".*/zero/zones/.*")
file(GLOB SIM_CHAOS zero/zones/chaos/*.cpp)
list(APPEND SIM_SOURCES ${SIM_CHAOS})
file(GLOB SIM_MAIN sim/*.cpp)
add_executable(duelsim ${SIM_SOURCES} ${SIM_MAIN})
target_compile_definitions(duelsim PRIVATE ZERO_SIM)
target_include_directories(duelsim PRIVATE . lib lib/glad/include lib/glfw/include)
if(WIN32)
  target_link_libraries(duelsim ws2_32)
else()
  target_link_libraries(duelsim dl -pthread)
endif()

# The simulator as a shared library for Python (ctypes): no main.
add_library(duelsim_lib SHARED ${SIM_SOURCES} ${SIM_MAIN})
set_target_properties(duelsim_lib PROPERTIES OUTPUT_NAME duelsim POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(duelsim_lib PRIVATE DUELSIM_LIBRARY ZERO_SIM)
target_include_directories(duelsim_lib PRIVATE . lib lib/glad/include lib/glfw/include)
if(WIN32)
  target_link_libraries(duelsim_lib ws2_32)
else()
  target_link_libraries(duelsim_lib dl -pthread)
endif()
