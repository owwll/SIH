#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <std_msgs/msg/string.hpp>

#include "navicore/types.hpp"
#include "navicore/fusion/eskf.hpp"
#include "navicore/calibration/auto_calib.hpp"

class NaviCoreRosNode : public rclcpp::Node {
public:
    NaviCoreRosNode() : Node("navicore_fusion_node") {
        RCLCPP_INFO(this->get_logger(), "Starting NaviCore AI ROS 2 Fusion Node...");

        eskf_ = std::make_unique<navicore::EskfFilter>();
        calibrator_ = std::make_unique<navicore::MountCalibrator>();

        // Subscribers
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/navicore/imu", 50,
            std::bind(&NaviCoreRosNode::ImuCallback, this, std::placeholders::_1));

        sub_gnss_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
            "/navicore/gps", 10,
            std::bind(&NaviCoreRosNode::GnssCallback, this, std::placeholders::_1));

        // Publishers
        pub_odom_ = this->create_publisher<nav_msgs::msg::Odometry>("/navicore/odom", 10);
        pub_pose_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/navicore/pose", 10);
        pub_status_ = this->create_publisher<std_msgs::msg::String>("/navicore/status", 10);
    }

private:
    std::unique_ptr<navicore::EskfFilter> eskf_;
    std::unique_ptr<navicore::MountCalibrator> calibrator_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sub_gnss_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_pose_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_status_;

    rclcpp::Time last_imu_time_{0, 0, RCL_ROS_TIME};

    void ImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        rclcpp::Time current_time = msg->header.stamp;
        float dt = 0.01f;
        if (last_imu_time_.nanoseconds() > 0) {
            dt = (current_time - last_imu_time_).seconds();
        }
        last_imu_time_ = current_time;

        navicore::ImuSample sample;
        sample.timestamp_nanos = current_time.nanoseconds();
        sample.ax = static_cast<float>(msg->linear_acceleration.x);
        sample.ay = static_cast<float>(msg->linear_acceleration.y);
        sample.az = static_cast<float>(msg->linear_acceleration.z);
        sample.gx = static_cast<float>(msg->angular_velocity.x);
        sample.gy = static_cast<float>(msg->angular_velocity.y);
        sample.gz = static_cast<float>(msg->angular_velocity.z);

        calibrator_->Ingest(sample);

        float vx, vy, vz;
        calibrator_->TransformBodyToVehicle(sample.ax, sample.ay, sample.az, vx, vy, vz);
        sample.ax = vx; sample.ay = vy; sample.az = vz;

        eskf_->Predict(sample, dt);

        PublishState(msg->header);
    }

    void GnssCallback(const sensor_msgs::msg::NavSatFix::SharedPtr msg) {
        if (msg->status.status >= sensor_msgs::msg::NavSatStatus::STATUS_FIX) {
            navicore::GnssFix fix;
            fix.timestamp_nanos = rclcpp::Time(msg->header.stamp).nanoseconds();
            fix.latitude_deg = msg->latitude;
            fix.longitude_deg = msg->longitude;
            fix.altitude_m = msg->altitude;
            eskf_->UpdateGnss(fix);
        }
    }

    void PublishState(const std_msgs::msg::Header& header) {
        navicore::FusionState state = eskf_->GetState();

        nav_msgs::msg::Odometry odom;
        odom.header.stamp = header.stamp;
        odom.header.frame_id = "map";
        odom.child_frame_id = "base_link";

        odom.pose.pose.position.x = state.latitude_deg;
        odom.pose.pose.position.y = state.longitude_deg;
        odom.pose.pose.position.z = state.altitude_m;
        odom.twist.twist.linear.x = state.speed_mps;

        pub_odom_->publish(odom);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<NaviCoreRosNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
