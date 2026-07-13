"""Launch two (or more) Insta360 cameras from ONE driver process.

A single `multicam` node opens every requested camera (one SDK init, a distinct
service port per camera) so they stream simultaneously without the cross-process
`bind: Address already in use` abort. One `decoder` node per camera namespace
turns the compressed stream into raw images (decoders don't touch the SDK, so
running several is fine).

Usage:
  ros2 launch insta360_ros_driver multicam.launch.py \
      cameras:="cam2:IAHYA2511DVC37,cam3:IAHYA25116W5XY"

Each "ns:serial" pair publishes under /<ns>:
  /<ns>/dual_fisheye/image/compressed   (from multicam)
  /<ns>/dual_fisheye/image              (from that namespace's decoder)
  /<ns>/imu/data_raw                    (from multicam)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


def _truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _parse_cameras(spec):
    """'cam2:SERIALA, cam3:SERIALB' -> (['cam2','cam3'], ['SERIALA','SERIALB'])."""
    namespaces, serials = [], []
    for pair in (p.strip() for p in spec.split(",")):
        if not pair:
            continue
        if ":" not in pair:
            raise RuntimeError(f"Bad 'cameras' entry '{pair}'; expected 'namespace:serial'.")
        ns, serial = pair.split(":", 1)
        ns, serial = ns.strip(), serial.strip()
        if not ns or not serial:
            raise RuntimeError(
                f"Bad 'cameras' entry '{pair}'; namespace and serial must be non-empty."
            )
        namespaces.append(ns)
        serials.append(serial)
    if not namespaces:
        raise RuntimeError("'cameras' is empty; provide at least one 'namespace:serial'.")
    return namespaces, serials


def _launch_setup(context, *args, **kwargs):
    spec = LaunchConfiguration("cameras").perform(context)
    video_resolution = LaunchConfiguration("video_resolution").perform(context)
    video_bitrate = int(LaunchConfiguration("video_bitrate").perform(context))
    base_service_port = int(LaunchConfiguration("base_service_port").perform(context))
    imu_filter = _truthy(LaunchConfiguration("imu_filter").perform(context))
    decoder = LaunchConfiguration("decoder").perform(context)

    namespaces, serials = _parse_cameras(spec)
    frame_prefixes = [ns + "_" for ns in namespaces]

    imu_config = os.path.join(
        get_package_share_directory("insta360_ros_driver"), "config", "imu_filter.yaml"
    )

    actions = [
        # One process owns every camera.
        Node(
            package="insta360_ros_driver",
            executable="multicam",
            name="insta360_multicam",
            output="screen",
            parameters=[
                {
                    "serials": serials,
                    "namespaces": namespaces,
                    "frame_prefixes": frame_prefixes,
                    "video_resolution": video_resolution,
                    "video_bitrate": video_bitrate,
                    "base_service_port": base_service_port,
                }
            ],
        ),
    ]

    # One decoder (+ optional IMU filter) per camera namespace.
    for ns, frame_prefix in zip(namespaces, frame_prefixes):
        group = [
            PushRosNamespace(ns),
            Node(
                package="insta360_ros_driver",
                executable="decoder",
                name="image_decoder",
                output="log",
                parameters=[
                    {
                        "compressed_topic": "dual_fisheye/image/compressed",
                        "uncompressed_topic": "dual_fisheye/image",
                        "frame_prefix": frame_prefix,
                        "skip_frame": 0,
                        "i_frame_only": False,
                        "decoder": decoder,
                    }
                ],
            ),
        ]
        if imu_filter:
            group.append(
                Node(
                    package="imu_filter_madgwick",
                    executable="imu_filter_madgwick_node",
                    name="imu_filter",
                    output="log",
                    parameters=[imu_config],
                )
            )
        actions.append(GroupAction(group))

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "cameras",
                default_value="cam2:IAHYA2511DVC37,cam3:IAHYA25116W5XY",
                description="Comma-separated 'namespace:serial' pairs, one per camera.",
            ),
            DeclareLaunchArgument("video_resolution", default_value="1920x960"),
            DeclareLaunchArgument("video_bitrate", default_value="524288"),
            DeclareLaunchArgument(
                "base_service_port",
                default_value="9999",
                description="SDK service port for camera 0; camera i uses base+i.",
            ),
            DeclareLaunchArgument("imu_filter", default_value="false"),
            DeclareLaunchArgument(
                "decoder",
                default_value="auto",
                description="H.264 decode backend: auto | cuvid | software | openh264. "
                "Latency: openh264 (1 frame) < software (2 frames) "
                "< cuvid (3 frames).",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
