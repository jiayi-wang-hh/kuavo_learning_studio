"""Dependency-free metadata for a selected observation; never reads a camera itself."""
import time
import uuid

TIMING_KEY = '_progress_monitor_observation'


def stamp_observation(selected_timestamps, ros_now, wall_now=None, clock_domain='ros'):
    cameras = {'observation.images.' + key: float(value) for key, value in selected_timestamps.items()
               if 'cam' in key and 'depth' not in key}
    # Oldest camera defines the conservative age of the three-camera observation.
    oldest = min(cameras.values()) if cameras else None
    return dict(observation_id=uuid.uuid4().hex,
                snapshot_wall_timestamp=time.time() if wall_now is None else float(wall_now),
                camera_timestamps=cameras, capture_clock_domain=clock_domain,
                frame_age_at_snapshot_ms=None if oldest is None or float(ros_now) < oldest else 1000 * (float(ros_now) - oldest))
