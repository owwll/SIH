cmake_minimum_required(VERSION 3.8)
project(navicore_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

include_directories(
  include
  ../core_cpp/include
)

add_executable(navicore_node
  src/navicore_node.cpp
  ../core_cpp/src/fusion/eskf.cpp
  ../core_cpp/src/calibration/auto_calib.cpp
)

ament_target_dependencies(navicore_node
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
)

install(TARGETS
  navicore_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
