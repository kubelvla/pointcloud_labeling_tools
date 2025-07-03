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


#include <omp.h>
#include <vector>
#include <fstream>
#include <chrono>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
using namespace std::chrono_literals;

class PointCloudLabelerNode : public rclcpp::Node {
public:
  PointCloudLabelerNode() : Node("pointcloud_labeler") {
    // Subscribers and publishers
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/hugin_raf_1/radar_data", 10,
      std::bind(&PointCloudLabelerNode::pointcloudCallback, this, std::placeholders::_1));

    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/labeled_cloud", 10);

    // prepare the TF2 buffer for reading tfs
    tfBuffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tfListener = std::make_unique<tf2_ros::TransformListener>(*tfBuffer);

    // Params
    world_frame_id_ = declare_parameter<std::string>("world_frame_id", "map");
    json_path = declare_parameter<std::string>("cuboid_file", "/home/vladimir/Downloads/Big_map_dense-v0.2.json");


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

    json j_labels = j["dataset"]["samples"][0]["labels"]["ground-truth"]["attributes"]["annotations"];

    for (const auto& item : j_labels) {
      Cuboid box;
      auto pos = item["position"];
      auto dim = item["dimensions"];
      auto rot = item["yaw"];

      box.center = Eigen::Vector3f(pos["x"], pos["y"], pos["z"]);
      box.extent = Eigen::Vector3f(dim["x"], dim["y"], dim["z"]);

      // Convert Euler angles (assumed XYZ) to rotation matrix
      Eigen::AngleAxisf rz(rot, Eigen::Vector3f::UnitZ());
      box.rotation = rz;

      box.id = item["category_id"];
      cuboids_.push_back(box);
    }

    RCLCPP_INFO(this->get_logger(), "Loaded %zu cuboids from %s", cuboids_.size(), json_path.c_str());
  }

  void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(*msg, pcl_cloud);
    std::vector<std::uint32_t> labels(pcl_cloud.size(), 0);

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


    // Parallel labeling
    //TODO: Enable the OMP pragma below. Breaks debugging symbols otherwise
//#pragma omp parallel for
    for (size_t i = 0; i < pcl_cloud.points.size(); ++i) {
      auto& pt = pcl_cloud.points[i];
      // convert to the world frame
      tf2::Vector3 tfd_point = stampedTransform * tf2::Vector3(pt.x, pt.y, pt.z);
      Eigen::Vector3f p(tfd_point.x(), tfd_point.y(), tfd_point.z());
      labels[i] = 0;

      for (const auto& cuboid : cuboids_) {
        if (isInsideCuboid(p, cuboid)) {
          labels[i] = cuboid.id;
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
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  std::vector<Cuboid> cuboids_;
  std::unique_ptr<tf2_ros::TransformListener> tfListener;
  std::unique_ptr<tf2_ros::Buffer> tfBuffer;
  std::string world_frame_id_;
  std::string json_path;

};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PointCloudLabelerNode>());
  rclcpp::shutdown();
  return 0;
}