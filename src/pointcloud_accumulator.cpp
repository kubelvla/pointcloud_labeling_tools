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
#include <pcl/io/pcd_io.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.hpp>

#include <filesystem>
#include <string>

enum class TriggerMode {
  MAX_DISTANCE,
  MAX_COUNT,
  NONE
};

namespace fs = std::filesystem;
std::string make_file_path(const std::string &folder, const std::string &filename)
{
  fs::path dir(folder);
  fs::path file(filename);
  fs::path full_path = dir / file;  // automatically inserts "/" if needed
  return full_path.string();        // or .string() / .u8string()
}

class PointCloudAccumulator : public rclcpp::Node {
public:
  PointCloudAccumulator() : Node("pointcloud_accumulator"),
                            tf_buffer_(this->get_clock()),
                            tf_listener_(tf_buffer_){
    declare_parameter<std::string>("input_topic", "/lidar/points");
    declare_parameter<std::string>("output_topic", "/assembled_cloud");
    declare_parameter<std::string>("fixed_frame", "odom");
    declare_parameter<double>("trigger_threshold", 1.0);
    declare_parameter<std::string>("trigger_mode", "max_count");
    declare_parameter<std::string>("output_frame", ""); // if left empty, kept in fixed frame
    declare_parameter<std::string>("save_output_folder", ""); // if left empty, no files saved
    declare_parameter<int>("skip_factor", 0); // how many point clouds should be skipped between accumulated ones


    get_parameter("input_topic", input_topic_);
    get_parameter("output_topic", output_topic_);
    get_parameter("fixed_frame", fixed_frame_);
    get_parameter("trigger_threshold", trigger_threshold_);
    get_parameter("output_frame", output_frame_);
    get_parameter("save_output_folder", output_folder_);
    get_parameter("skip_factor", skip_factor_);
    skip_counter_ = 0;

    std::string selected_mode;
    get_parameter("trigger_mode", selected_mode);

    if(selected_mode == "max_distance") {
      trigger_mode_ = TriggerMode::MAX_DISTANCE;
    } else if(selected_mode == "max_count") {
      trigger_mode_ = TriggerMode::MAX_COUNT;
    } else {
      RCLCPP_ERROR(this->get_logger(), "Unknown trigger mode, shutting down.");
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
    // Check if we skip
    if (skip_counter_<skip_factor_) {
      skip_counter_++;
      return;
    }else {
      skip_counter_ = 0;
    }

    // Lookup TF for putting the incoming cloud into a fixed frame
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
      initial_time_ = msg->header.stamp;
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
    std::vector<double> current_pose = {transform.transform.translation.x,
                                        transform.transform.translation.y,
                                        transform.transform.translation.z};  // Keep this for the filename

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
        out_msg = transformed_out_msg;

        cloud_pub_->publish(*out_msg);
        RCLCPP_INFO(this->get_logger(), "Published an accum. point cloud in the %s frame made of %d clouds after travelling %f meters.",
                    output_frame_.c_str(), accum_count_, accum_distance_);
      }

      if (!output_folder_.empty()) {    // We will attempt to save the accumulated output also as a pcd file
        std::stringstream ss;

        ss << "cloud_" << initial_time_->sec << "." << std::setw(9) << std::setfill('0')
           << initial_time_->nanosec << "_to_" << msg->header.stamp.sec << "." << std::setw(9)
           << msg->header.stamp.nanosec << "_at_" << std::fixed << std::setprecision(1)
           << current_pose[0] << "_" << current_pose[1] << "_" << current_pose[2] << "_" << ".pcd";

        std::string full_path = make_file_path(output_folder_, ss.str());

        pcl::PCLPointCloud2::Ptr saved_pcl(new pcl::PCLPointCloud2());
        pcl_conversions::toPCL(*out_msg, *saved_pcl);

        pcl::PCDWriter writer;
        //writer.writeASCII(full_path, *saved_pcl);
        writer.writeBinary(full_path, *saved_pcl);
      }

      accumulated_cloud_.reset(new pcl::PCLPointCloud2()); // Reset cloud
      previous_pose_.reset(); // Reset distance tracker
      initial_time_.reset(); // Clear the time as well
    }
  }

  // Parameters
  std::string input_topic_;
  std::string output_topic_;
  std::string fixed_frame_;
  std::string output_frame_;
  std::string output_folder_;
  double trigger_threshold_;
  TriggerMode trigger_mode_;
  double accum_distance_;
  int accum_count_;
  int skip_factor_;
  int skip_counter_;


  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  pcl::PCLPointCloud2::Ptr accumulated_cloud_;
  std::optional<geometry_msgs::msg::Vector3> previous_pose_;
  std::optional<builtin_interfaces::msg::Time> initial_time_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PointCloudAccumulator>());
  rclcpp::shutdown();
  return 0;
}