// Shared streaming helpers for the Insta360 ROS driver, used by BOTH the
// single-camera node (main.cpp) and the single-process multi-camera node
// (multicam.cpp). Keeping the StreamDelegate + resolution handling here means
// the two nodes publish byte-for-byte identical messages.
//
// NOTE on the IMU timestamp anchor: it is stored as PER-INSTANCE members
// (t0_sdk_ms_/t0_ros_), NOT function-local statics. The multi-camera node
// owns one delegate per camera; a shared static anchor would make the second
// camera's IMU stream inherit the first camera's epoch and corrupt its stamps.
#pragma once

#include <map>
#include <string>
#include <vector>
#include <chrono>
#include <memory>

#include <camera/camera.h>
#include <camera/photography_settings.h>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/imu.hpp"

namespace insta360 {

// Human-readable "WIDTHxHEIGHT" -> SDK resolution enum. The actual delivered
// resolution is negotiated per camera model over USB and may differ from the
// requested value (e.g. the X5 USB live stream is effectively capped at
// ~2656x1328 regardless of what is requested here).
inline const std::map<std::string, ins_camera::VideoResolution>& ResolutionMap() {
    static const std::map<std::string, ins_camera::VideoResolution> kMap = {
        {"3840x1920", ins_camera::VideoResolution::RES_3840_1920P30},
        {"2880x2880", ins_camera::VideoResolution::RES_2880_2880P30},
        {"2560x1280", ins_camera::VideoResolution::RES_2560_1280P30},
        {"2304x1152", ins_camera::VideoResolution::RES_1152_1152P30}, // 2304x1152 @30
        {"1920x960",  ins_camera::VideoResolution::RES_1920_960P30},
        {"1440x720",  ins_camera::VideoResolution::RES_1440_720P30},
    };
    return kMap;
}

// Fallback used when the requested resolution is rejected by the camera.
constexpr ins_camera::VideoResolution kFallbackResolution =
    ins_camera::VideoResolution::RES_1920_960P30;
constexpr const char* kFallbackResolutionStr = "1920x960";

inline std::string CameraTypeToString(ins_camera::CameraType type) {
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

inline std::string SupportedResolutionList() {
    std::string out;
    for (const auto& kv : ResolutionMap()) {
        if (!out.empty()) out += ", ";
        out += kv.first;
    }
    return out;
}

// Publishes the H.264 compressed dual-fisheye stream + 6-DOF IMU for a single
// camera, on topics RELATIVE to its node's namespace. Launching the owning node
// under /cam3 publishes /cam3/dual_fisheye/image/compressed and
// /cam3/imu/data_raw, letting several cameras coexist on one ROS graph.
class TestStreamDelegate : public ins_camera::StreamDelegate {
private:
    std::shared_ptr<rclcpp::Node> node_;
    std::string frame_prefix_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;

    // Per-instance IMU clock anchor (see file header). One epoch per camera.
    int64_t t0_sdk_ms_ = -1;
    rclcpp::Time t0_ros_{0, 0, RCL_ROS_TIME};

public:
    TestStreamDelegate(const std::shared_ptr<rclcpp::Node>& node,
                       const std::string& frame_prefix = "")
        : node_(node), frame_prefix_(frame_prefix) {
        compressed_pub_ = node_->create_publisher<sensor_msgs::msg::CompressedImage>(
            "dual_fisheye/image/compressed", rclcpp::QoS(10));
        // The SDK delivers gyro in bursts of ~50 samples (the X5 live stream's
        // ~500 Hz batched into ~10 callbacks/sec). The default SensorDataQoS
        // depth of 5 can only hold ~5 of each burst before the subscriber drains
        // it, dropping ~90% in transit (subscribers then see only ~50 Hz). Keep
        // best-effort semantics but deepen the queue to hold several full bursts
        // so the full ~500 Hz survives. Consumers must subscribe with a
        // similarly deep queue to receive it all.
        imu_pub_ = node_->create_publisher<sensor_msgs::msg::Imu>(
            "imu/data_raw", rclcpp::SensorDataQoS().keep_last(200));
        RCLCPP_INFO(node_->get_logger(),
            "Publisher for compressed images and IMU created.");
    }

    virtual ~TestStreamDelegate() {}

    void OnAudioData(const uint8_t* /*data*/, size_t /*size*/, int64_t /*timestamp*/) override {}

    void OnVideoData(const uint8_t* data, size_t size, int64_t /*timestamp*/,
                     uint8_t /*streamType*/, int stream_index) override {
        // We only care about the main video stream (index 0).
        if (stream_index == 0 && size > 0 && compressed_pub_) {
            auto msg = std::make_unique<sensor_msgs::msg::CompressedImage>();
            msg->header.stamp = node_->get_clock()->now();
            msg->header.frame_id = frame_prefix_ + "camera_frame";
            // The subscriber needs this to select the correct decoder.
            msg->format = "h264";
            msg->data.assign(data, data + size);
            compressed_pub_->publish(std::move(msg));
        }
    }

    void OnGyroData(const std::vector<ins_camera::GyroData>& data) override {
        // Use the SDK's per-sample timestamp (milliseconds since some epoch the
        // SDK doesn't document) instead of ros::now(). The SDK delivers IMU
        // samples in bursts over USB; stamping with ros::now() would give every
        // sample in a burst nearly the same time, making the log useless for
        // Allan-variance / frequency analysis.
        //
        // Anchor the SDK clock to wall-clock at the first sample so rosbag
        // timestamps look like real time; subsequent samples are spaced by their
        // SDK delta (uniform 1-2 ms at the X5's ~500 Hz live stream).
        for (const auto& gyro : data) {
            if (t0_sdk_ms_ < 0) {
                t0_sdk_ms_ = gyro.timestamp;
                t0_ros_ = node_->get_clock()->now();
            }
            const int64_t dt_ns =
                static_cast<int64_t>(gyro.timestamp - t0_sdk_ms_) * 1'000'000;
            const rclcpp::Time stamp =
                t0_ros_ + rclcpp::Duration(std::chrono::nanoseconds(dt_ns));

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

            for (int i = 0; i < 9; i++) {
                msg->angular_velocity_covariance[i] = 0;
                msg->linear_acceleration_covariance[i] = 0;
            }
            imu_pub_->publish(std::move(msg));
        }
    }

    void OnExposureData(const ins_camera::ExposureData& /*data*/) override {}
};

// Resolve a "WIDTHxHEIGHT" request to the SDK enum, warning + falling back to
// kFallbackResolution if it is unknown. `resolved_str_out` receives the string
// that was actually selected.
inline ins_camera::VideoResolution ResolveResolution(
        const std::string& requested, std::string& resolved_str_out,
        rclcpp::Logger logger) {
    const auto& map = ResolutionMap();
    auto it = map.find(requested);
    if (it != map.end()) {
        resolved_str_out = requested;
        return it->second;
    }
    RCLCPP_WARN(logger,
        "Unknown video_resolution '%s'; using default '%s'. Supported: %s",
        requested.c_str(), kFallbackResolutionStr, SupportedResolutionList().c_str());
    resolved_str_out = kFallbackResolutionStr;
    return kFallbackResolution;
}

// Start live streaming on an already-opened camera, retrying once at the
// known-good fallback resolution if the requested one is rejected. On success
// `resolved_str_out` holds the resolution that actually started. Returns false
// if streaming could not be started at all.
inline bool StartLiveStreamingWithFallback(
        const std::shared_ptr<ins_camera::Camera>& cam,
        const std::string& requested_res, int video_bitrate,
        rclcpp::Logger logger, std::string& resolved_str_out) {
    std::string res_str;
    ins_camera::VideoResolution resolution =
        ResolveResolution(requested_res, res_str, logger);

    ins_camera::LiveStreamParam param;
    param.video_resolution = resolution;
    param.lrv_video_resulution = ins_camera::VideoResolution::RES_1440_720P30;
    param.video_bitrate = static_cast<uint32_t>(video_bitrate);
    param.enable_audio = false;
    param.using_lrv = false;

    RCLCPP_INFO(logger, "Requesting live stream at %s (bitrate %d bps).",
        res_str.c_str(), video_bitrate);

    if (cam->StartLiveStreaming(param)) {
        resolved_str_out = res_str;
        return true;
    }

    // The requested resolution may be unsupported on this model. Retry once with
    // the known-good fallback before giving up.
    if (resolution != kFallbackResolution) {
        RCLCPP_WARN(logger,
            "Failed to start live streaming at %s; retrying at fallback %s.",
            res_str.c_str(), kFallbackResolutionStr);
        param.video_resolution = kFallbackResolution;
        if (cam->StartLiveStreaming(param)) {
            resolved_str_out = kFallbackResolutionStr;
            return true;
        }
    }
    RCLCPP_ERROR(logger,
        "Failed to start live streaming. The requested resolution may be "
        "unsupported on this camera model over USB.");
    return false;
}

} // namespace insta360
