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

    // Per-instance VIDEO clock anchor: the SDK delivers a device-side per-AU
    // timestamp (ms since camera boot), which is capture-side and immune to
    // USB/SDK delivery jitter. Anchored separately from the IMU because the
    // two callbacks' epochs are not documented to match.
    //
    // The anchor is the MINIMUM observed (host_arrival - device_ts) offset
    // over the first kVideoAnchorWindow AUs, then frozen. A single-AU
    // snapshot anchor made the eps constant (docs/camera_latency_check.md)
    // vary ~15 ms across relaunches - one AU's delivery latency is a per-
    // launch lottery; the min over ~3 s of AUs converges to the fastest
    // delivery path, repeatable to a few ms. This also inherently ignores
    // the stale mid-GOP tap-in AU the stream starts with (its delivery lag
    // is large, so it never wins the min).
    static constexpr uint32_t kVideoAnchorWindow = 90;  // ~3 s at 30 fps
    int64_t video_anchor_offset_ns_ = 0;  // min(host_ns - device_ns), 0 = unset
    int64_t video_prev_sdk_ms_ = -1;
    int64_t video_last_stamp_ns_ = 0;
    uint32_t video_au_count_ = 0;
    bool video_delta_logged_ = false;
    bool video_anchor_logged_ = false;

    // Map an SDK video timestamp to a ROS stamp: device time plus the frozen
    // min-delay anchor offset. Falls back to arrival time when the SDK
    // provides no usable timestamp; re-anchors from scratch if the device
    // clock regresses (stream restart). Published stamps are clamped
    // monotonic - anchor refinement during the window can only move the
    // mapping earlier, and a >one-frame refinement step could otherwise
    // produce a backwards stamp.
    rclcpp::Time VideoStamp(int64_t sdk_ts_ms) {
        if (sdk_ts_ms <= 0) {
            return node_->get_clock()->now();
        }
        const int64_t device_ns = sdk_ts_ms * 1'000'000;
        // Device clock regression = stream restart: start a fresh anchor window.
        if (video_prev_sdk_ms_ > 0 && sdk_ts_ms < video_prev_sdk_ms_) {
            video_anchor_offset_ns_ = 0;
            video_au_count_ = 0;
            video_anchor_logged_ = false;
        }
        ++video_au_count_;
        if (video_au_count_ <= kVideoAnchorWindow) {
            const int64_t offset_ns =
                node_->get_clock()->now().nanoseconds() - device_ns;
            if (video_anchor_offset_ns_ == 0 || offset_ns < video_anchor_offset_ns_) {
                video_anchor_offset_ns_ = offset_ns;
            }
        } else if (!video_anchor_logged_) {
            RCLCPP_INFO(node_->get_logger(),
                "video stamp anchor locked after %u AUs (min-delay offset).",
                kVideoAnchorWindow);
            video_anchor_logged_ = true;
        }
        // One-shot unit sanity check on a steady-state AU pair (~33 ms at
        // 30 fps confirms the SDK video timestamps are in milliseconds like
        // the gyro ones). Deliberately not the first pair: the stream taps
        // in mid-GOP, so the first delta is an encoder-restart gap (~434 ms
        // observed), not the cadence.
        if (!video_delta_logged_ && video_au_count_ == 31 && video_prev_sdk_ms_ > 0) {
            RCLCPP_INFO(node_->get_logger(),
                "video SDK timestamp cadence: %lld ms between consecutive AUs "
                "(expect ~33 at 30 fps)",
                static_cast<long long>(sdk_ts_ms - video_prev_sdk_ms_));
            video_delta_logged_ = true;
        }
        video_prev_sdk_ms_ = sdk_ts_ms;
        int64_t stamp_ns = video_anchor_offset_ns_ + device_ns;
        if (stamp_ns <= video_last_stamp_ns_) {
            stamp_ns = video_last_stamp_ns_ + 1;  // keep stamps strictly monotonic
        }
        video_last_stamp_ns_ = stamp_ns;
        return rclcpp::Time(stamp_ns, RCL_ROS_TIME);
    }

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

    void OnVideoData(const uint8_t* data, size_t size, int64_t timestamp,
                     uint8_t /*streamType*/, int stream_index) override {
        // We only care about the main video stream (index 0).
        if (stream_index == 0 && size > 0 && compressed_pub_) {
            auto msg = std::make_unique<sensor_msgs::msg::CompressedImage>();
            // Capture-side device timestamp (anchored to wall-clock), NOT
            // arrival time: this keeps header.stamp uniform across USB/SDK
            // delivery jitter. The decoder propagates this stamp through to
            // the decoded image topic.
            msg->header.stamp = VideoStamp(timestamp);
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

// SDK 2.1.1 example flow for X4/X5: switch the camera into its dedicated
// live-view sub-mode and set the preview resolution for the live-stream
// function mode BEFORE StartLiveStreaming. The 2.1.1 example claims this makes
// X5 preview resolutions selectable (3840x1920 / 2560x1280 / 2160x1080 /
// 1920x960) where the plain flow is fixed at 2656x1328. On failure we warn
// and continue with the plain flow.
inline void ApplyLiveViewMode(const std::shared_ptr<ins_camera::Camera>& cam,
                              ins_camera::VideoResolution resolution,
                              rclcpp::Logger logger) {
    if (!cam->SetVideoSubMode(ins_camera::SubVideoMode::VIDEO_LIVEVIEW)) {
        RCLCPP_WARN(logger,
            "SetVideoSubMode(VIDEO_LIVEVIEW) failed; continuing without live-view mode.");
        return;
    }
    ins_camera::RecordParams record_params;
    record_params.resolution = resolution;
    record_params.bitrate = 0;  // example uses 0; stream bitrate comes from LiveStreamParam
    if (!cam->SetVideoCaptureParams(
            record_params, ins_camera::CameraFunctionMode::FUNCTION_MODE_LIVE_STREAM)) {
        RCLCPP_WARN(logger,
            "SetVideoCaptureParams(FUNCTION_MODE_LIVE_STREAM) failed; continuing.");
        return;
    }
    RCLCPP_INFO(logger,
        "live-view mode applied (VIDEO_LIVEVIEW + FUNCTION_MODE_LIVE_STREAM).");
}

// Select which physical sensor(s) feed the pipeline ("front" / "rear" /
// "all"). Empty string = leave the camera as-is (the only safe value on X5).
//
// WARNING - VERIFIED HARMFUL ON X5 (fw 1.1.22, 2026-07-13): sending
// SetActiveSensor over USB drops the USB session instantly ("The device has
// been disconnected"), the SDK call times out after ~13 s and then returns a
// BOGUS success, and the camera resets out of Android USB mode (apparent
// firmware reboot; requires re-selecting Android mode + replug to recover).
// Kept only for experiments on other models/firmwares. The single-lens
// latency win is NOT reachable this way on X5; crop downstream instead.
inline void ApplyActiveSensor(const std::shared_ptr<ins_camera::Camera>& cam,
                              const std::string& which, rclcpp::Logger logger) {
    if (which.empty()) return;
    ins_camera::SensorDevice dev;
    if (which == "front")      dev = ins_camera::SensorDevice::SENSOR_DEVICE_FRONT;
    else if (which == "rear")  dev = ins_camera::SensorDevice::SENSOR_DEVICE_REAR;
    else if (which == "all")   dev = ins_camera::SensorDevice::SENSOR_DEVICE_ALL;
    else {
        RCLCPP_WARN(logger,
            "Unknown active_sensor '%s' (expected front|rear|all|empty); leaving camera as-is.",
            which.c_str());
        return;
    }
    if (cam->SetActiveSensor(dev)) {
        RCLCPP_INFO(logger, "SetActiveSensor(%s) succeeded.", which.c_str());
    } else {
        RCLCPP_WARN(logger, "SetActiveSensor(%s) failed; camera keeps its current sensor mode.",
            which.c_str());
    }
}

// Log the camera's self-reported preview parameters after streaming starts.
// encode_type matters: the decoder assumes h264, so a live-view-mode flip to
// h265 must be caught here. delay_timestamp is undocumented but by name may be
// the camera's own preview-delay estimate; sweep_time is the rolling-shutter
// readout time.
inline void LogPreviewParam(const std::shared_ptr<ins_camera::Camera>& cam,
                            rclcpp::Logger logger) {
    const auto pp = cam->GetPreviewParam();
    RCLCPP_INFO(logger,
        "camera preview param: encode_type=%d (h264 expected) delay_timestamp=%lld "
        "sweep_time=%lld",
        static_cast<int>(pp.encode_type),
        static_cast<long long>(pp.delay_timestamp),
        static_cast<long long>(pp.sweep_time));
}

// Start live streaming on an already-opened camera, retrying once at the
// known-good fallback resolution if the requested one is rejected. On success
// `resolved_str_out` holds the REQUESTED resolution the SDK accepted - NOT the
// actual frame size, which the camera sets and may differ (the X5 caps the live
// stream to 2656x1328 regardless; the decoder logs the true size). Returns false
// if streaming could not be started at all.
inline bool StartLiveStreamingWithFallback(
        const std::shared_ptr<ins_camera::Camera>& cam,
        const std::string& requested_res, int video_bitrate,
        rclcpp::Logger logger, std::string& resolved_str_out,
        bool using_lrv = false, bool live_view_mode = false) {
    std::string res_str;
    ins_camera::VideoResolution resolution =
        ResolveResolution(requested_res, res_str, logger);

    ins_camera::LiveStreamParam param;
    param.video_resolution = resolution;
    param.lrv_video_resulution = ins_camera::VideoResolution::RES_1440_720P30;
    param.video_bitrate = static_cast<uint32_t>(video_bitrate);
    param.enable_audio = false;
    // using_lrv=true REPLACES the main preview with the camera's low-res
    // stream (1024x512 per the SDK demo) - a latency experiment: smaller
    // encode may shrink the camera-side encode buffer. When active, the LRV
    // bitrate is the operative knob, so mirror the requested bitrate into it.
    param.using_lrv = using_lrv;
    if (using_lrv) {
        param.lrv_video_bitrate = static_cast<uint32_t>(video_bitrate);
        RCLCPP_INFO(logger,
            "using_lrv enabled: camera will substitute its low-res preview "
            "stream (expect ~1024x512; see the decoder's 'actual resolution' log).");
    }

    if (live_view_mode) {
        ApplyLiveViewMode(cam, resolution, logger);
    }

    RCLCPP_INFO(logger, "Requesting live stream at %s (bitrate %d bps).",
        res_str.c_str(), video_bitrate);

    if (cam->StartLiveStreaming(param)) {
        resolved_str_out = res_str;
        LogPreviewParam(cam, logger);
        return true;
    }

    // The requested resolution may be unsupported on this model. Retry once with
    // the known-good fallback before giving up.
    if (resolution != kFallbackResolution) {
        RCLCPP_WARN(logger,
            "Failed to start live streaming at %s; retrying at fallback %s.",
            res_str.c_str(), kFallbackResolutionStr);
        if (live_view_mode) {
            ApplyLiveViewMode(cam, kFallbackResolution, logger);
        }
        param.video_resolution = kFallbackResolution;
        if (cam->StartLiveStreaming(param)) {
            resolved_str_out = kFallbackResolutionStr;
            LogPreviewParam(cam, logger);
            return true;
        }
    }
    RCLCPP_ERROR(logger,
        "Failed to start live streaming. The requested resolution may be "
        "unsupported on this camera model over USB.");
    return false;
}

} // namespace insta360
