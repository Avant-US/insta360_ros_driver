// Single-process, multi-camera Insta360 driver node.
//
// WHY THIS EXISTS: the desktop CameraSDK binds a fixed local service port when a
// live stream is started. Running one driver PROCESS per camera makes the second
// process abort with `bind: Address already in use` (SIGABRT). The robust fix is
// ONE process that opens every camera, giving each a distinct service port via
// Camera::SetServicePort() before Open(). A single GetAvailableDevices() call
// also avoids the concurrent-discovery race, and a clean StopLiveStreaming()+
// Close() on shutdown avoids leaving USB interfaces in LIBUSB_ERROR_BUSY.
//
// Each camera gets its own rclcpp::Node under its own namespace (e.g. /cam2,
// /cam3) and its own StreamDelegate, so topics are identical to the single-camera
// node: /<ns>/dual_fisheye/image/compressed and /<ns>/imu/data_raw. Run a
// `decoder` node per namespace (see multicam.launch.py) to get the raw images.
//
// Parameters (parallel arrays, one entry per camera):
//   serials          (string[])  required, SDK serial numbers to open
//   namespaces       (string[])  required, one ROS namespace per serial
//   frame_prefixes   (string[])  optional, TF frame prefix per camera
//                                 (defaults to "<namespace>_")
//   video_resolution (string)    requested "WxH" (default 1920x960)
//   video_bitrate    (int)       bits/sec (default 524288)
//   base_service_port(int)       first SDK service port; camera i uses base+i
//                                 (default 9999)

#include <algorithm>
#include <ctime>
#include <memory>
#include <string>
#include <vector>

#include <camera/camera.h>
#include <camera/device_discovery.h>
#include <camera/photography_settings.h>

#include "rclcpp/rclcpp.hpp"

#include "insta360_stream.hpp"

namespace {

// Holds everything we must keep alive for one streaming camera.
struct CameraSession {
    std::string serial;
    std::string ns;
    std::shared_ptr<rclcpp::Node> node;
    std::shared_ptr<ins_camera::Camera> cam;
    std::shared_ptr<ins_camera::StreamDelegate> delegate;
};

// rclcpp namespaces must be absolute ("/cam2"). Accept "cam2" or "/cam2".
std::string NormalizeNamespace(const std::string& ns) {
    if (ns.empty()) return "/";
    return (ns.front() == '/') ? ns : ("/" + ns);
}

} // namespace

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);

    // A config-only node owns the parameters that describe the camera fleet.
    auto cfg = rclcpp::Node::make_shared("insta360_multicam");
    const auto serials =
        cfg->declare_parameter<std::vector<std::string>>("serials", std::vector<std::string>{});
    const auto namespaces =
        cfg->declare_parameter<std::vector<std::string>>("namespaces", std::vector<std::string>{});
    const auto frame_prefixes =
        cfg->declare_parameter<std::vector<std::string>>("frame_prefixes", std::vector<std::string>{});
    const std::string video_resolution =
        cfg->declare_parameter<std::string>("video_resolution", "1920x960");
    const int video_bitrate =
        cfg->declare_parameter<int>("video_bitrate", 1024 * 1024 / 2);
    const int base_service_port =
        cfg->declare_parameter<int>("base_service_port", 9999);

    auto logger = cfg->get_logger();

    if (serials.empty()) {
        RCLCPP_ERROR(logger, "No 'serials' provided; nothing to stream.");
        rclcpp::shutdown();
        return -1;
    }
    if (serials.size() != namespaces.size()) {
        RCLCPP_ERROR(logger,
            "'serials' (%zu) and 'namespaces' (%zu) must have the same length.",
            serials.size(), namespaces.size());
        rclcpp::shutdown();
        return -1;
    }

    // Discover every connected camera ONCE.
    ins_camera::DeviceDiscovery discovery;
    auto list = discovery.GetAvailableDevices();
    if (list.empty()) {
        RCLCPP_ERROR(logger, "No available camera devices found.");
        discovery.FreeDeviceDescriptors(list);
        rclcpp::shutdown();
        return -1;
    }
    std::string available;
    for (const auto& d : list) {
        if (!available.empty()) available += ", ";
        available += d.serial_number;
    }
    RCLCPP_INFO(logger, "Discovered %zu camera(s): [%s].", list.size(), available.c_str());

    std::vector<CameraSession> sessions;
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(cfg);

    for (size_t i = 0; i < serials.size(); ++i) {
        const std::string& serial = serials[i];
        const std::string ns = NormalizeNamespace(namespaces[i]);
        const std::string frame_prefix =
            (i < frame_prefixes.size() && !frame_prefixes[i].empty())
                ? frame_prefixes[i]
                : (namespaces[i] + "_");

        // Find the requested serial among the discovered devices.
        int chosen = -1;
        for (size_t j = 0; j < list.size(); ++j) {
            if (list[j].serial_number == serial) { chosen = static_cast<int>(j); break; }
        }
        if (chosen < 0) {
            RCLCPP_ERROR(logger,
                "Requested serial '%s' (namespace %s) not among connected cameras: [%s]. Skipping.",
                serial.c_str(), ns.c_str(), available.c_str());
            continue;
        }

        const auto& device = list[chosen];
        auto cam = std::make_shared<ins_camera::Camera>(device.info);

        // Distinct service port per camera so two SDK sessions in this one
        // process do not collide on the default port (the cross-process
        // "bind: Address already in use" failure mode).
        const int service_port = base_service_port + static_cast<int>(i);
        cam->SetServicePort(service_port);

        if (!cam->Open()) {
            RCLCPP_ERROR(logger,
                "Failed to open camera (serial %s, service port %d). Skipping.",
                serial.c_str(), service_port);
            continue;
        }
        RCLCPP_INFO(logger,
            "Opened %s (serial %s, fw %s) -> namespace %s, service port %d.",
            insta360::CameraTypeToString(device.camera_type).c_str(),
            device.serial_number.c_str(), device.fw_version.c_str(),
            ns.c_str(), service_port);

        // Per-camera node + delegate. Topics are relative -> namespaced.
        auto cam_node = std::make_shared<rclcpp::Node>("insta360_camera", ns);
        cam->SyncLocalTimeToCamera(static_cast<uint64_t>(time(NULL)), 0);

        std::shared_ptr<ins_camera::StreamDelegate> delegate =
            std::make_shared<insta360::TestStreamDelegate>(cam_node, frame_prefix);
        cam->SetStreamDelegate(delegate);

        std::string resolved_str;
        if (!insta360::StartLiveStreamingWithFallback(
                cam, video_resolution, video_bitrate, cam_node->get_logger(),
                resolved_str)) {
            RCLCPP_ERROR(logger, "Camera %s failed to start streaming. Skipping.",
                serial.c_str());
            cam->Close();
            continue;
        }
        RCLCPP_INFO(logger,
            "[%s] Live streaming started (requested %s; actual frame size is set by "
            "the camera - the X5 is capped, see the decoder's 'actual resolution' log).",
            ns.c_str(), resolved_str.c_str());

        exec.add_node(cam_node);
        sessions.push_back(CameraSession{serial, ns, cam_node, cam, delegate});
    }

    discovery.FreeDeviceDescriptors(list);

    if (sessions.empty()) {
        RCLCPP_ERROR(logger, "No cameras started; exiting.");
        rclcpp::shutdown();
        return -1;
    }
    RCLCPP_INFO(logger, "Streaming %zu camera(s) in one process.", sessions.size());

    // Spin until SIGINT (rclcpp's default handler triggers shutdown -> returns).
    exec.spin();

    // Clean teardown: stop streams first (stops delegate callbacks), then close
    // each camera so the SDK releases its USB interface (no stuck BUSY).
    RCLCPP_INFO(logger, "Shutting down; stopping %zu camera(s).", sessions.size());
    for (auto& s : sessions) {
        if (s.cam) s.cam->StopLiveStreaming();
    }
    for (auto& s : sessions) {
        if (s.cam) s.cam->Close();
    }

    rclcpp::shutdown();
    return 0;
}
