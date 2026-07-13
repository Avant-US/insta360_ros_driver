#include <ctime>
#include <memory>
#include <string>
#include <vector>

#include <camera/camera.h>
#include <camera/device_discovery.h>
#include <camera/photography_settings.h>

#include "rclcpp/rclcpp.hpp"

#include "insta360_stream.hpp"

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
        // Resolution is configurable at launch as "WIDTHxHEIGHT" (see the map in
        // insta360_stream.hpp). The actual delivered resolution is negotiated by
        // the camera over USB and may differ from the request (X5 is capped).
        const std::string requested_res =
            node_->declare_parameter<std::string>("video_resolution", "1920x960");
        const int video_bitrate =
            node_->declare_parameter<int>("video_bitrate", 1024 * 1024 / 2);
        // Substitute the camera's low-res (~1024x512) preview stream for the
        // main one - a camera-side latency experiment (see insta360_stream.hpp).
        const bool using_lrv =
            node_->declare_parameter<bool>("using_lrv", false);
        // SDK 2.1.1 X4/X5 live-view flow (SetVideoSubMode + SetVideoCaptureParams
        // before streaming) - may make X5 preview resolution selectable.
        const bool live_view_mode =
            node_->declare_parameter<bool>("live_view_mode", false);
        // Sensor selection: "front" | "rear" | "all" | "" (leave as-is).
        // Single-sensor halves the encoded pixels (see insta360_stream.hpp).
        const std::string active_sensor =
            node_->declare_parameter<std::string>("active_sensor", "");

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
            insta360::CameraTypeToString(device.camera_type).c_str(),
            device.serial_number.c_str(),
            device.fw_version.c_str());
        discovery.FreeDeviceDescriptors(list);

        insta360::ApplyActiveSensor(cam, active_sensor, node_->get_logger());

        std::shared_ptr<ins_camera::StreamDelegate> delegate =
            std::make_shared<insta360::TestStreamDelegate>(node_, frame_prefix);
        cam->SetStreamDelegate(delegate);

        const auto start = time(NULL);
        cam->SyncLocalTimeToCamera(static_cast<uint64_t>(start), 0 /*no UTC offset*/);

        std::string resolved_str;
        if (!insta360::StartLiveStreamingWithFallback(
                cam, requested_res, video_bitrate, node_->get_logger(),
                resolved_str, using_lrv, live_view_mode)) {
            return -1;
        }
        RCLCPP_INFO(node_->get_logger(),
            "Live streaming started (requested %s; actual frame size is set by the "
            "camera - the X5 is capped, see the decoder's 'actual resolution' log).",
            resolved_str.c_str());
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
