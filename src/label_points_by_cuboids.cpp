#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/point_cloud.h"
#include "pointcloud_labeling_tools/cuboid.hpp"
#include <tf2_ros/transform_listener.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <unordered_set>


#include "pointcloud_labeling_tools/srv/label_pcd.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <omp.h>
#include <vector>
#include <fstream>
#include <chrono>
#include <nlohmann/json.hpp>
#include <array>
#include <cmath>

using json = nlohmann::json;
using namespace std::chrono_literals;

class PointCloudLabelerNode : public rclcpp::Node {
public:
  PointCloudLabelerNode() : Node("pointcloud_labeler") {
    // Subscribers and publishers and services
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "input_pointcloud", 10,
      std::bind(&PointCloudLabelerNode::pointcloudCallback, this, std::placeholders::_1));

    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("labeled_cloud", 10);
    cuboid_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("labeled_cuboids", 10);

    label_service_ = this->create_service<pointcloud_labeling_tools::srv::LabelPCD>(
      "label_pcd_file",
    std::bind(&PointCloudLabelerNode::labelPCDCallback, this, std::placeholders::_1, std::placeholders::_2)
);

    // prepare the TF2 buffer for reading tfs
    tfBuffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tfListener = std::make_unique<tf2_ros::TransformListener>(*tfBuffer);

    // Params
    world_frame_id_ = declare_parameter<std::string>("world_frame_id", "map");
    json_path = declare_parameter<std::string>("cuboid_file", "/home/vladimir/Downloads/Big_map_dense-v0.2.json");
    sample_id = declare_parameter<int>("sample_id", 0);
    label_set_name = declare_parameter<std::string>("label_set_name", "ground-truth");
    priority_list = this->declare_parameter<std::vector<int>>("label_priorities", {3,7,1,5,4,6,2});
    for (size_t i = 0; i < priority_list.size(); ++i) {
      label_priority_map[static_cast<int>(priority_list[i])] = static_cast<int>(i);
    }

    loadCuboids();
  }

private:
  void loadCuboids() {
    std::ifstream in(json_path);
    if (!in) {
      RCLCPP_ERROR(this->get_logger(), "Could not open cuboid JSON file: %s", json_path.c_str());
      return;
    }

    json j;
    in >> j;
    json j_labels;

    try {
      j_labels = j["dataset"]["samples"][sample_id]["labels"][label_set_name]["attributes"]["annotations"];
    }catch (std::exception &ex) {
      RCLCPP_ERROR(this->get_logger(), "Failed to extract annotations from the json file: %s", ex.what());
      std::exit(1);
    }

    if (j_labels.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to extract any annotations from the json file.");
      std::exit(1);
    }

    for (const auto& item : j_labels) {
      Cuboid box;
      auto pos = item["position"];
      auto dim = item["dimensions"];
      auto rot = item["rotation"];

      box.center = Eigen::Vector3f(pos["x"], pos["y"], pos["z"]);
      box.extent = Eigen::Vector3f(dim["x"], dim["y"], dim["z"]);

      // Convert quaternion into a rotation matrix
      Eigen::Quaternionf q(rot["qw"], rot["qx"], rot["qy"], rot["qz"]);
      box.rotation = q;
      box.id = item["id"];
      box.category_id = item["category_id"];
      cuboids_.push_back(box);
    }

    RCLCPP_INFO(this->get_logger(), "Loaded %zu cuboids from %s", cuboids_.size(), json_path.c_str());
  }

  std_msgs::msg::ColorRGBA colorFromId(uint32_t category_id) {
    std_msgs::msg::ColorRGBA color;

    // Use golden angle to generate distinct hues
    float hue = std::fmod(0.61803398875f * category_id, 1.0f);  // golden ratio
    float saturation = 0.7f;
    float value = 0.9f;

    float h = hue * 6.0f;
    int i = static_cast<int>(std::floor(h));
    float f = h - i;
    float p = value * (1.0f - saturation);
    float q = value * (1.0f - saturation * f);
    float t = value * (1.0f - saturation * (1.0f - f));

    switch (i % 6) {
      case 0: color.r = value; color.g = t;     color.b = p;     break;
      case 1: color.r = q;     color.g = value; color.b = p;     break;
      case 2: color.r = p;     color.g = value; color.b = t;     break;
      case 3: color.r = p;     color.g = q;     color.b = value; break;
      case 4: color.r = t;     color.g = p;     color.b = value; break;
      case 5: color.r = value; color.g = p;     color.b = q;     break;
    }

    color.a = 0.2f;  // semi-transparent
    return color;
  }

  void visualize_cuboids(const std::unordered_set<int> &used_cuboid_ids) {
    visualization_msgs::msg::MarkerArray marker_array;
    int marker_id = 0;

    for (const auto& cuboid : cuboids_) {
      if (used_cuboid_ids.count(cuboid.id) == 0)
        continue;

      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = world_frame_id_;
      marker.header.stamp = this->now();
      marker.ns = "labeled_cuboids";
      marker.id = marker_id++;
      marker.type = visualization_msgs::msg::Marker::CUBE;
      marker.action = visualization_msgs::msg::Marker::ADD;

      // Position
      marker.pose.position.x = cuboid.center.x();
      marker.pose.position.y = cuboid.center.y();
      marker.pose.position.z = cuboid.center.z();

      // Orientation
      Eigen::Quaternionf q(cuboid.rotation);
      marker.pose.orientation.x = q.x();
      marker.pose.orientation.y = q.y();
      marker.pose.orientation.z = q.z();
      marker.pose.orientation.w = q.w();

      // Size
      marker.scale.x = cuboid.extent.x();
      marker.scale.y = cuboid.extent.y();
      marker.scale.z = cuboid.extent.z();

      // Color (optional: based on label)
      marker.color = colorFromId(cuboid.category_id);
      marker.lifetime = rclcpp::Duration(10.0s);  // Persist until updated
      marker_array.markers.push_back(marker);
    }

    cuboid_pub_->publish(marker_array);
  }

  void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(*msg, pcl_cloud);
    std::vector<std::uint32_t> labels(pcl_cloud.size(), 0); // 0 is default for no label - background
    std::vector<int> label_ranks(pcl_cloud.size(), std::numeric_limits<int>::max()); // for tracking label priority

    // Look up transform from the point cloud to the world frame
    geometry_msgs::msg::TransformStamped transform;
    tf2::Stamped<tf2::Transform> stampedTransform;

    try{
      transform = tfBuffer->lookupTransform(world_frame_id_,
                                            msg->header.frame_id,
                                            msg->header.stamp,
                                            rclcpp::Duration(0.5s));
    }
    catch(tf2::TransformException &ex){
      RCLCPP_ERROR(this->get_logger(), "Point cloud TF lookup failed because: %s", ex.what());
      return;
    }

    // convert to a format suitable for transforming points
    tf2::convert(transform, stampedTransform);

    // track what cuboids we have used
    std::unordered_set<int> used_cuboid_ids;

    // Parallel labeling
    //TODO: Enable the OMP pragma below. But it breaks debugging symbols
//#pragma omp parallel for
    for (size_t i = 0; i < pcl_cloud.points.size(); ++i) {
      auto& pt = pcl_cloud.points[i];
      // convert to the world frame
      tf2::Vector3 tfd_point = stampedTransform * tf2::Vector3(pt.x, pt.y, pt.z);
      Eigen::Vector3f p(tfd_point.x(), tfd_point.y(), tfd_point.z());

      // check all cuboids (TODO: can be optimized, limiting the cuboids or the points on some proximity rules)
      labels[i] = 0;
      for (const auto& cuboid : cuboids_) {
        if (isInsideCuboid(p, cuboid)) {

          // check priority
          int priority = label_priority_map.count(cuboid.category_id) > 0 ? label_priority_map[cuboid.category_id] : 9999;  // default low priority
          if (priority < label_ranks[i]) {
            labels[i] = cuboid.category_id;
            label_ranks[i] = priority;
          }
//#pragma omp critical
          used_cuboid_ids.insert(cuboid.id);
          break;
        }
      }
    }

    // Create a copy of the original message
    sensor_msgs::msg::PointCloud2 out_msg = *msg; // copy header and metadata
    // Add a new data field
    sensor_msgs::msg::PointField extra_id_field;
    extra_id_field.name = "label_id";
    extra_id_field.offset = out_msg.point_step;
    extra_id_field.datatype = sensor_msgs::msg::PointField::UINT8;
    extra_id_field.count = 1;
    out_msg.fields.push_back(extra_id_field);

    out_msg.point_step += 1;
    out_msg.row_step = out_msg.point_step * out_msg.width;
    out_msg.data.resize(out_msg.row_step * out_msg.height);

    // Copy the original data fields
    const uint8_t* src_ptr = msg->data.data();
    uint8_t* dst_ptr = out_msg.data.data();

    for (size_t i = 0; i < pcl_cloud.points.size(); ++i) {
      std::memcpy(dst_ptr, src_ptr, msg->point_step);
      src_ptr += msg->point_step;
      dst_ptr += out_msg.point_step;
    }

    // Fill in the label id's
    sensor_msgs::PointCloud2Iterator<std::uint8_t> label_iter(out_msg, "label_id");
    for (size_t i = 0; i < labels.size(); ++i, ++label_iter) {
      *label_iter = labels[i];
    }

    // Publish the labelled point cloud
    pub_->publish(out_msg);

    visualize_cuboids(used_cuboid_ids);
  }

  void labelPCDCallback(
    const std::shared_ptr<pointcloud_labeling_tools::srv::LabelPCD::Request> request,
    std::shared_ptr<pointcloud_labeling_tools::srv::LabelPCD::Response> response)
  {
    const std::string& input_path = request->file_path;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile(input_path, *cloud) < 0) {
      response->success = false;
      response->message = "Failed to load file: " + input_path;
      return;
    }

    // Run labeling logic here...
    std::vector<uint32_t> labels(cloud->size(), 0);
    std::vector<int> label_ranks(cloud->size(), std::numeric_limits<int>::max());

    for (size_t i = 0; i < cloud->points.size(); ++i) {
      Eigen::Vector3f pt(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
      for (const auto& cuboid : cuboids_) {
        if (!isInsideCuboid(pt, cuboid)) continue;

        int priority = label_priority_map.count(cuboid.category_id) > 0 ? label_priority_map[cuboid.category_id] : 9999;

        if (priority < label_ranks[i]) {
          label_ranks[i] = priority;
          labels[i] = cuboid.category_id;
        }
      }
    }

    // Add label field and save output
    pcl::PointCloud<pcl::PointXYZL>::Ptr labeled(new pcl::PointCloud<pcl::PointXYZL>);
    labeled->resize(cloud->size());

    for (size_t i = 0; i < cloud->size(); ++i) {
      labeled->points[i].x = cloud->points[i].x;
      labeled->points[i].y = cloud->points[i].y;
      labeled->points[i].z = cloud->points[i].z;
      labeled->points[i].label = labels[i];
    }

    // Construct output filename
    std::filesystem::path in_path(input_path);
    std::string out_path = in_path.parent_path().string() + "/labeled_" + in_path.filename().string();

    if (pcl::io::savePCDFileBinary(out_path, *labeled) < 0) {
      response->success = false;
      response->message = "Failed to save labeled file to: " + out_path;
      return;
    }

    response->success = true;
    response->message = "Labeled PCD saved to: " + out_path;
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr cuboid_pub_;
  rclcpp::Service<pointcloud_labeling_tools::srv::LabelPCD>::SharedPtr label_service_;
  std::vector<Cuboid> cuboids_;
  std::unique_ptr<tf2_ros::TransformListener> tfListener;
  std::unique_ptr<tf2_ros::Buffer> tfBuffer;
  std::string world_frame_id_;
  std::string json_path;
  int sample_id;
  std::string label_set_name;
  std::vector<long> priority_list;
  std::unordered_map<int, int> label_priority_map;

};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PointCloudLabelerNode>());
  rclcpp::shutdown();
  return 0;
}