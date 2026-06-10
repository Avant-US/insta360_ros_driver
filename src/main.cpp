#include <iostream>
#include <thread>
#include <string>
#include <vector>
#include <atomic>
#include <map>
#include <algorithm>

#include <camera/camera.h>
#include <camera/photography_settings.h>
#include <camera/device_discovery.h>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/imu.hpp"

namespace {

// Human-readable "WIDTHxHEIGHT" -> SDK resolution enum. Note that the actual
// delivered resolution is negotiated per camera model over USB and may differ
// from the requested value (e.g. the X5 USB live stream is effectively capped
// at ~2656x1328 regardless of what is requested here).
const std::map<std::string, ins_camera::VideoResolution> kResolutionMap = {
    {"3840x1920", ins_camera::VideoResolution::RES_3840_1920P30},
    {"2880x2880", ins_camera::VideoResolution::RES_2880_2880P30},
    {"2560x1280", ins_camera::VideoResolution::RES_2560_1280P30},
    {"2304x1152", ins_camera::VideoResolution::RES_1152_1152P30}, // 2304x1152 @30
    {"1920x960",  ins_camera::VideoResolution::RES_1920_960P30},
    {"1440x720",  ins_camera::VideoResolution::RES_1440_720P30},
};

// Fallback used when the requested resolution is rejected by the camera.
constexpr ins_camera::VideoResolution kFallbackResolution =
    ins_camera::VideoResolution::RES_1920_960P30;
constexpr const char* kFallbackResolutionStr = "1920x960";

std::string CameraTypeToString(ins_camera::CameraType type) {
    switch (type) {
        case ins_camera::CameraType::Insta360OneX:   return "ONE X";
        case ins_camera::CameraType::Insta360OneR:   return "ONE R";
        case ins_camera::CameraType::Insta360OneRS:  return "ONE RS";
        case ins_camera::CameraType::Insta360OneX2:  return "X2";
        case ins_camera::CameraType::Insta360X3:     return "X3";
        case ins_camera::CameraType::Insta360X4:     return "X4";
        case ins_camera::CameraType::Insta360X5:     return "X5";
        case ins_camera::CameraType::Insta360X4Air:  return "X4 Air";
        default:                                     return "Unknown";
    }
}

std::string SupportedResolutionList() {
    std::string out;
    for (const auto& kv : kResolutionMap) {
        if (!out.empty()) out += ", ";
        out += kv.first;
    }
    return out;
}

} // namespace

class TestStreamDelegate : public ins_camera::StreamDelegate {
private:
    std::shared_ptr<rclcpp::Node> node_;
    std::string frame_prefix_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;

public:
    TestStreamDelegate(const std::shared_ptr<rclcpp::Node>& node,
                       const std::string& frame_prefix = "")
        : node_(node), frame_prefix_(frame_prefix) {
        // Relative topic names (no leading '/') so the node's namespace prefixes
        // them: launching this node under /cam3 publishes /cam3/dual_fisheye/...
        // and /cam3/imu/data_raw, letting two cameras coexist on one ROS graph.
        compressed_pub_ = node_->create_publisher<sensor_msgs::msg::CompressedImage>(
            "dual_fisheye/image/compressed",
            rclcpp::QoS(10)
        );

        // Publisher for IMU data
        imu_pub_ = node_->create_publisher<sensor_msgs::msg::Imu>("imu/data_raw", rclcpp::SensorDataQoS());
        RCLCPP_INFO(node_->get_logger(), "Publisher for compressed images and IMU created.");
    }

    virtual ~TestStreamDelegate() {}

    void OnAudioData(const uint8_t* data, size_t size, int64_t timestamp) override {}

    void OnVideoData(const uint8_t* data, size_t size, int64_t timestamp, uint8_t streamType, int stream_index) override {
        // We only care about the main video stream (index 0)
        if (stream_index == 0 && size > 0 && compressed_pub_) {
            auto msg = std::make_unique<sensor_msgs::msg::CompressedImage>();

            // Set the header
            msg->header.stamp = node_->get_clock()->now();
            msg->header.frame_id = frame_prefix_ + "camera_frame";

            // Set the format to H.264
            // The subscriber will need to know this to select the correct decoder.
            msg->format = "h264";

            // Copy the compressed video data directly into the message
            msg->data.assign(data, data + size);

            compressed_pub_->publish(std::move(msg));
        }
    }

    void OnGyroData(const std::vector<ins_camera::GyroData>& data) override {
        // Use the SDK's per-sample timestamp (milliseconds since some epoch
        // that the SDK doesn't document) instead of ros::now(). The SDK
        // delivers IMU samples in bursts over USB; if we stamp with ros::now
        // every sample in a burst gets nearly the same time and the resulting
        // log is useless for Allan-variance / frequency analysis.
        //
        // Anchor the SDK clock to wall-clock at the first sample so rosbag
        // timestamps look like real time; subsequent samples are spaced by
        // their SDK delta (uniform 1-2 ms at the X5's ~500 Hz live stream).
        static int64_t t0_sdk_ms = -1;
        static rclcpp::Time t0_ros = rclcpp::Time(0, 0, RCL_ROS_TIME);
        for (const auto& gyro : data) {
            if (t0_sdk_ms < 0) {
                t0_sdk_ms = gyro.timestamp;
                t0_ros = node_->get_clock()->now();
            }
            const int64_t dt_ns =
                static_cast<int64_t>(gyro.timestamp - t0_sdk_ms) * 1'000'000;
            const rclcpp::Time stamp =
                t0_ros + rclcpp::Duration(std::chrono::nanoseconds(dt_ns));

            auto msg = std::make_unique<sensor_msgs::msg::Imu>();
            msg->header.stamp = stamp;
            msg->header.frame_id = frame_prefix_ + "imu_frame";
            msg->angular_velocity.x = gyro.gx;
            msg->angular_velocity.y = gyro.gy;
            msg->angular_velocity.z = gyro.gz;
            
            msg->linear_acceleration.x = gyro.ax * 9.80665;
            msg->linear_acceleration.y = gyro.ay * 9.80665;
            msg->linear_acceleration.z = gyro.az * 9.80665;

            msg->orientation.x = 0.0;
            msg->orientation.y = 0.0;
            msg->orientation.z = 0.0;
            msg->orientation.w = 1.0; // Neutral orientation
            msg->orientation_covariance[0] = -1.0; // No orientation data available

            for (int i = 0; i < 9; i++)
            {
                msg->angular_velocity_covariance[i] = 0;
                msg->linear_acceleration_covariance[i] = 0;
            }
            imu_pub_->publish(std::move(msg));
        }
    }

    void OnExposureData(const ins_camera::ExposureData& data) override {}
};

class CameraWrapper {
private:
    std::shared_ptr<ins_camera::Camera> cam;
    std::shared_ptr<rclcpp::Node> node_;

public:
    CameraWrapper(const std::shared_ptr<rclcpp::Node>& node) : node_(node) {}

    ~CameraWrapper() {
        if (cam) {
            cam->Close();
        }
    }

    int run_camera() {
        // `serial` selects a specific camera by its SDK serial number when more
        // than one X5 is plugged in (empty = first discovered). `frame_prefix`
        // is prepended to the published frame_ids so two cameras don't collide in
        // a shared TF tree (e.g. "cam3_" -> cam3_camera_frame / cam3_imu_frame).
        const std::string serial =
            node_->declare_parameter<std::string>("serial", "");
        const std::string frame_prefix =
            node_->declare_parameter<std::string>("frame_prefix", "");

        ins_camera::DeviceDiscovery discovery;
        auto list = discovery.GetAvailableDevices();
        if (list.empty()) {
            RCLCPP_ERROR(node_->get_logger(), "No available camera devices found.");
            return -1;
        }

        // Pick the requested serial, or the first device when none is requested.
        int chosen = -1;
        if (serial.empty()) {
            chosen = 0;
        } else {
            for (size_t i = 0; i < list.size(); ++i) {
                if (list[i].serial_number == serial) { chosen = static_cast<int>(i); break; }
            }
            if (chosen < 0) {
                std::string available;
                for (const auto& d : list) {
                    if (!available.empty()) available += ", ";
                    available += d.serial_number;
                }
                RCLCPP_ERROR(node_->get_logger(),
                    "Requested serial '%s' not among connected cameras: [%s].",
                    serial.c_str(), available.c_str());
                discovery.FreeDeviceDescriptors(list);
                return -1;
            }
        }

        const auto& device = list[chosen];
        cam = std::make_shared<ins_camera::Camera>(device.info);
        if (!cam->Open()) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to open camera (serial %s).",
                device.serial_number.c_str());
            discovery.FreeDeviceDescriptors(list);
            return -1;
        }
        RCLCPP_INFO(node_->get_logger(),
            "Camera opened successfully. Type: %s, Serial: %s, Firmware: %s",
            CameraTypeToString(device.camera_type).c_str(),
            device.serial_number.c_str(),
            device.fw_version.c_str());
        discovery.FreeDeviceDescriptors(list);

        std::shared_ptr<ins_camera::StreamDelegate> delegate =
            std::make_shared<TestStreamDelegate>(node_, frame_prefix);
        cam->SetStreamDelegate(delegate);

        auto start = time(NULL);

        uint64_t utc_time = static_cast<uint64_t>(start);
        uint32_t offset_time = 0; //no offset from UTC

        cam->SyncLocalTimeToCamera(utc_time,offset_time);

        // Resolution is configurable at launch as "WIDTHxHEIGHT" (see kResolutionMap).
        // The actual delivered resolution is negotiated by the camera over USB and
        // may differ from the request (e.g. the X5 is effectively capped).
        const std::string requested_res =
            node_->declare_parameter<std::string>("video_resolution", "1920x960");
        const int video_bitrate =
            node_->declare_parameter<int>("video_bitrate", 1024 * 1024 / 2);

        ins_camera::VideoResolution resolution = kFallbackResolution;
        std::string resolution_str = kFallbackResolutionStr;
        auto it = kResolutionMap.find(requested_res);
        if (it != kResolutionMap.end()) {
            resolution = it->second;
            resolution_str = requested_res;
        } else {
            RCLCPP_WARN(node_->get_logger(),
                "Unknown video_resolution '%s'; using default '%s'. Supported: %s",
                requested_res.c_str(), kFallbackResolutionStr,
                SupportedResolutionList().c_str());
        }

        ins_camera::LiveStreamParam param;
        param.video_resolution = resolution;
        param.lrv_video_resulution = ins_camera::VideoResolution::RES_1440_720P30;
        param.video_bitrate = static_cast<uint32_t>(video_bitrate);
        param.enable_audio = false;
        param.using_lrv = false;

        RCLCPP_INFO(node_->get_logger(),
            "Requesting live stream at %s (bitrate %d bps).",
            resolution_str.c_str(), video_bitrate);

        if (!cam->StartLiveStreaming(param)) {
            // The requested resolution may be unsupported on this model. Retry once
            // with the known-good fallback before giving up.
            if (resolution != kFallbackResolution) {
                RCLCPP_WARN(node_->get_logger(),
                    "Failed to start live streaming at %s; retrying at fallback %s.",
                    resolution_str.c_str(), kFallbackResolutionStr);
                param.video_resolution = kFallbackResolution;
                if (cam->StartLiveStreaming(param)) {
                    RCLCPP_INFO(node_->get_logger(),
                        "Live streaming started at fallback %s.", kFallbackResolutionStr);
                    return 0;
                }
            }
            RCLCPP_ERROR(node_->get_logger(),
                "Failed to start live streaming. The requested resolution may be "
                "unsupported on this camera model over USB.");
            return -1;
        }

        RCLCPP_INFO(node_->get_logger(), "Live streaming started at %s.",
            resolution_str.c_str());
        return 0;
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("insta_publisher");
    
    CameraWrapper camera(node);
    if (camera.run_camera() != 0) {
        rclcpp::shutdown();
        return -1;
    }
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}