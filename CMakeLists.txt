cmake_minimum_required(VERSION 3.8)
project(pointcloud_labeling_tools)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)


add_definitions(${PCL_DEFINITIONS})


rosidl_generate_interfaces(${PROJECT_NAME}
        "srv/LabelPCD.srv"
)

ament_export_dependencies(rosidl_default_runtime)

include_directories(
        include
        ${PCL_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)


add_executable(label_points_by_cuboids src/label_points_by_cuboids.cpp)
ament_target_dependencies(label_points_by_cuboids rclcpp sensor_msgs pcl_conversions tf2 tf2_ros tf2_geometry_msgs visualization_msgs)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")    #This is to use our own service, otherwise the compiler won't find the headers
target_link_libraries(label_points_by_cuboids ${PCL_LIBRARIES} "${cpp_typesupport_target}")
target_compile_features(label_points_by_cuboids PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_compile_options(label_points_by_cuboids PRIVATE -fopenmp)
target_link_libraries(label_points_by_cuboids gomp nlohmann_json::nlohmann_json)  # OpenMP runtime if needed


add_executable(pointcloud_accumulator src/pointcloud_accumulator.cpp)
ament_target_dependencies(pointcloud_accumulator rclcpp sensor_msgs tf2_ros tf2_geometry_msgs pcl_conversions pcl_ros)
target_link_libraries(pointcloud_accumulator ${PCL_LIBRARIES})


install(TARGETS label_points_by_cuboids pointcloud_accumulator
  DESTINATION lib/${PROJECT_NAME})

install(
        DIRECTORY launch
        DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
