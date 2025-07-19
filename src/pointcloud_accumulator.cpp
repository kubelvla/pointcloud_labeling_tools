#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/common/io.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.hpp>

enum class TriggerMode {
  MAX_DISTANCE,
  MAX_COUNT,
  NONE
};

class PointCloudAccumulator : public rclcpp::Node {
public:
  PointCloudAccumulator() : Node("pointcloud_accumulator"),
                            tf_buffer_(this->get_clock()),
                            tf_listener_(tf_buffer_) {
    declare_parameter<std::string>("input_topic", "/lidar/points");
    declare_parameter<std::string>("output_topic", "/assembled_cloud");
    declare_parameter<std::string>("fixed_frame", "odom");
    declare_parameter<double>("trigger_threshold", 1.0);
    declare_parameter<std::string>("trigger_mode", "max_count");
    declare_parameter<std::string>("output_frame", ""); // if left empty, kept in fixed frame

    get_parameter("input_topic", input_topic_);
    get_parameter("output_topic", output_topic_);
    get_parameter("fixed_frame", fixed_frame_);
    get_parameter("trigger_threshold", trigger_threshold_);
    get_parameter("output_frame", output_frame_);

    std::string selected_mode;
    get_parameter("trigger_mode", selected_mode);

    if(selected_mode == "max_distance") {
      trigger_mode_ = TriggerMode::MAX_DISTANCE;
    } else if(selected_mode == "max_count") {
      trigger_mode_ = TriggerMode::MAX_COUNT;
    } else {
      RCLCPP_ERROR(this->get_logger(), "Uknown trigger mode, shutting down.");
      std::exit(1);
    }

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic_, 10,
        std::bind(&PointCloudAccumulator::cloudCallback, this, std::placeholders::_1));

    cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, 10);

    accumulated_cloud_ = std::make_shared<pcl::PCLPointCloud2>();

  }

private:
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // Lookup TF as before
    geometry_msgs::msg::TransformStamped transform;
    try {
      transform = tf_buffer_.lookupTransform(fixed_frame_, msg->header.frame_id,
                                             tf2::TimePoint(std::chrono::seconds(msg->header.stamp.sec) +
                                                            std::chrono::nanoseconds(msg->header.stamp.nanosec)),
                                             tf2::durationFromSec(0.2));
    } catch (tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s", ex.what());
      return;
    }

    // Transform input msg cloud to fixed frame
    auto transformed_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
    pcl_ros::transformPointCloud(fixed_frame_, transform, *msg, *transformed_msg);

    // Turn into the PCL equivalent for easier concatenation
    pcl::PCLPointCloud2::Ptr transformed_pcl(new pcl::PCLPointCloud2());
    pcl_conversions::toPCL(*transformed_msg, *transformed_pcl);

    // Accumulate
    if (accumulated_cloud_->width == 0) {
      *accumulated_cloud_ = *transformed_pcl;
    } else {
      pcl::concatenate(*accumulated_cloud_, *transformed_pcl, *accumulated_cloud_);
    }

    // Compute distance traveled
    if (!previous_pose_.has_value()) {
      previous_pose_ = transform.transform.translation;
      accum_distance_ = 0.0;
      accum_count_ = 0;
    }

    double dx = transform.transform.translation.x - previous_pose_->x;
    double dy = transform.transform.translation.y - previous_pose_->y;
    double dz = transform.transform.translation.z - previous_pose_->z;
    double distance = std::sqrt(dx*dx + dy*dy + dz*dz);

    accum_distance_ += distance;
    accum_count_++;

    if( (trigger_mode_ == TriggerMode::MAX_DISTANCE && accum_distance_ >= trigger_threshold_) ||
        (trigger_mode_ == TriggerMode::MAX_COUNT  && accum_count_ >= (int)trigger_threshold_) )
    {
      auto out_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
      pcl_conversions::fromPCL(*accumulated_cloud_, *out_msg);
      out_msg->header.stamp = msg->header.stamp;

      if(output_frame_.empty()) {                // we keep the accumulated point cloud in the fixed frame
        out_msg->header.frame_id = fixed_frame_;
        cloud_pub_->publish(*out_msg);
        RCLCPP_INFO(this->get_logger(), "Published an accum. point cloud made of %d clouds after travelling %f meters.", accum_count_, accum_distance_);

      }else{                                     // we want to put into some other frame before publishing
        try {
          transform = tf_buffer_.lookupTransform(output_frame_, fixed_frame_,
                                                 tf2::TimePoint(std::chrono::seconds(msg->header.stamp.sec) +
                                                                std::chrono::nanoseconds(msg->header.stamp.nanosec)),
                                                 tf2::durationFromSec(0.2));
        } catch (tf2::TransformException &ex) {
          RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s, dropping the accumulated cloud", ex.what());
          accumulated_cloud_.reset(new pcl::PCLPointCloud2()); // Reset cloud
          previous_pose_.reset(); // Reset distance tracker
          return;
        }

        auto transformed_out_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        pcl_ros::transformPointCloud(output_frame_, transform, *out_msg, *transformed_out_msg);

        cloud_pub_->publish(*out_msg);
        RCLCPP_INFO(this->get_logger(), "Published an accum. point cloud in the %s frame made of %d clouds after travelling %f meters.",
                    output_frame_.c_str(), accum_count_, accum_distance_);
      }

      accumulated_cloud_.reset(new pcl::PCLPointCloud2()); // Reset cloud
      previous_pose_.reset(); // Reset distance tracker
    }
  }

  // Parameters
  std::string input_topic_;
  std::string output_topic_;
  std::string fixed_frame_;
  std::string output_frame_;
  double trigger_threshold_;
  TriggerMode trigger_mode_;
  double accum_distance_;
  int accum_count_;


  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  pcl::PCLPointCloud2::Ptr accumulated_cloud_;
  std::optional<geometry_msgs::msg::Vector3> previous_pose_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PointCloudAccumulator>());
  rclcpp::shutdown();
  return 0;
}