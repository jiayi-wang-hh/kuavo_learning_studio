from __future__ import annotations

from io import BytesIO
import time

from .models import CameraFrame
from .observation import ObservationSynchronizer, ros_stamp_seconds


def decode_compressed_rgb(data):
    """Decode a ROS CompressedImage JPEG/PNG without the NumPy-ABI-sensitive cv_bridge."""
    from PIL import Image

    with Image.open(BytesIO(bytes(data))) as image:
        return image.convert("RGB").copy()


class KuavoSensorBridge:
    """Read-only ROS bridge. This class never creates a publisher or motion service client."""
    def __init__(self, config: dict, synchronizer: ObservationSynchronizer, tactile_processor):
        import rospy
        from sensor_msgs.msg import CompressedImage, JointState
        try:
            from kuavo_msgs.msg import dexhandTouchState
        except ImportError:
            from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import dexhandTouchState

        self.rospy, self.synchronizer = rospy, synchronizer
        self.subscribers = []
        topics = config["ros"]["topics"]
        camera_map = {
            "head": topics.get("head_rgb"),
            "left_wrist": topics.get("left_wrist_rgb") if config["cameras"].get("left_wrist", {}).get("enabled") else None,
            "right_wrist": topics.get("right_wrist_rgb") if config["cameras"].get("right_wrist", {}).get("enabled") else None,
        }
        for name, topic in camera_map.items():
            if topic and name in synchronizer.buffers:
                self.subscribers.append(rospy.Subscriber(topic, CompressedImage, self._image_cb(name), queue_size=1, buff_size=2**20))
        tactile_topic = topics.get("tactile")
        if tactile_topic:
            self.subscribers.append(rospy.Subscriber(tactile_topic, dexhandTouchState, self._tactile_cb(tactile_processor), queue_size=1))
        state_topic = topics.get("robot_state")
        if state_topic:
            try:
                from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import sensorsData
            except ImportError as exc:
                raise RuntimeError("Kuavo sensorsData message is unavailable; source the ROS/SDK environment before live dry-run") from exc
            self.subscribers.append(rospy.Subscriber(state_topic, sensorsData, self._sensor_state_cb, queue_size=1))
        joint_state_topic = topics.get("joint_state")
        if joint_state_topic:
            self.subscribers.append(rospy.Subscriber(joint_state_topic, JointState, self._joint_cb, queue_size=1))

    def _image_cb(self, name):
        def callback(msg):
            try:
                image = decode_compressed_rgb(msg.data)
                stamp, source = ros_stamp_seconds(msg.header.stamp)
                self.synchronizer.push_image(CameraFrame(name, image, stamp, time.monotonic(), source))
            except Exception as exc:
                self.rospy.logwarn_throttle(5.0, "Ignoring invalid %s image: %s", name, exc)
        return callback

    def _tactile_cb(self, processor):
        def callback(msg):
            self.synchronizer.tactile = processor.process(msg)
            self.synchronizer.tactile_received_at = time.monotonic()
        return callback

    def _joint_cb(self, msg):
        positions = list(msg.position)
        self.synchronizer.robot_state.update({
            "joint_state_valid": bool(positions), "joint_names": list(msg.name),
            "joint_positions": positions, "joint_state_stamp": msg.header.stamp.to_sec(),
            "valid": bool(positions), "safety_ok": None, "motion_state": "unknown",
            "gripper_state_valid": False, "received_at": time.monotonic(),
        })

    def _sensor_state_cb(self, msg):
        try:
            positions = list(msg.joint_data.joint_q)
        except (AttributeError, TypeError):
            self.rospy.logwarn_throttle(5.0, "Invalid Kuavo sensorsData robot-state message")
            return
        self.synchronizer.robot_state.update({
            "joint_state_valid": len(positions) >= 14, "joint_positions": positions,
            "joint_state_stamp": msg.header.stamp.to_sec(),
            "valid": len(positions) >= 14, "safety_ok": None,
            "motion_state": "unknown", "gripper_state_valid": False,
            "received_at": time.monotonic(),
        })

    def close(self):
        for sub in self.subscribers:
            sub.unregister()
        self.subscribers.clear()


class KuavoControlSignalBridge:
    """Read-only pause/stop control signals shared with the Kuavo eval scripts.

    A stop request is latched for this process. These topics are operator requests,
    not acknowledgements that the robot has physically stopped.
    """
    def __init__(self, config: dict):
        import threading
        import rospy
        from std_msgs.msg import Bool

        self.rospy = rospy
        self.paused = threading.Event()
        self.stop_requested = threading.Event()
        topics = config.get("ros", {}).get("topics", {})
        pause_topic = topics.get("pause_state", "/kuavo/pause_state")
        stop_topic = topics.get("stop_state", "/kuavo/stop_state")
        self.subscribers = [
            rospy.Subscriber(pause_topic, Bool, self._pause_cb, queue_size=10),
            rospy.Subscriber(stop_topic, Bool, self._stop_cb, queue_size=10),
        ]

    def _pause_cb(self, msg):
        if bool(msg.data):
            self.paused.set()
        else:
            self.paused.clear()

    def _stop_cb(self, msg):
        if bool(msg.data):
            self.stop_requested.set()

    def close(self):
        for sub in self.subscribers:
            sub.unregister()
        self.subscribers.clear()

