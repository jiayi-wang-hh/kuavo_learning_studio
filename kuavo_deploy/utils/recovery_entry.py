"""Read-only bag reference and bounded right-arm entry trajectory; no ROS/SDK on import."""
from __future__ import annotations
import math
from pathlib import Path


def validate_entry_config(config):
    if type(config.get('enabled')) is not bool:
        raise ValueError('recovery_right_arm_entry.enabled must be boolean')
    if not config['enabled']:
        return
    if not Path(config['bag_path']).is_file():
        raise FileNotFoundError(config['bag_path'])
    for key in ['duration', 'rate_hz', 'max_velocity', 'max_acceleration', 'arrival_tolerance',
                'left_hold_tolerance', 'arrival_timeout', 'stable_seconds']:
        value = config.get(key)
        if type(value) not in [int, float] or not math.isfinite(value) or value <= 0:
            raise ValueError(f'recovery_right_arm_entry.{key} must be positive and finite')
    if not 10 <= config['rate_hz'] <= 100 or config['stable_seconds'] >= config['arrival_timeout']:
        raise ValueError('Entry requires 10-100Hz and stable_seconds < arrival_timeout')


def joints(values, size):
    result = [float(x) for x in values]
    if len(result) != size or not all(math.isfinite(x) for x in result):
        raise ValueError(f'Expected {size} finite joint values')
    return result


def measured_arm_reference(message, slices):
    if len(slices) != 2 or any(len(s) != 2 or s[1] - s[0] != 7 or s[0] < 0 for s in slices):
        raise ValueError('Expected left and right seven-joint slices in measured radians')
    raw = message.joint_data.joint_q
    if max(s[1] for s in slices) > len(raw):
        raise ValueError('Bag joint vector is shorter than configured arm slices')
    return joints([q for start, end in slices for q in raw[start:end]], 14)


def load_bag_right_reference(path, topic, slices, lower, upper):
    import rosbag  # Read-only. No ROS node, publisher, or model initialization.
    with rosbag.Bag(str(path)) as bag:
        for _, message, stamp in bag.read_messages(topics=[topic]):
            if getattr(message, '_type', None) != 'kuavo_msgs/sensorsData':
                raise ValueError('Entry bag topic must contain kuavo_msgs/sensorsData measured radians')
            arm = measured_arm_reference(message, slices)
            right = arm[7:]
            lo, hi = joints(lower, 14)[7:], joints(upper, 14)[7:]
            if any(not low <= q <= high for q, low, high in zip(right, lo, hi)):
                raise ValueError('Reference right arm is outside configured joint limits')
            return dict(bag_path=str(Path(path).resolve()), topic=topic, bag_time_s=stamp.to_sec(),
                        source='first measured sensorsData message; radians; configured arm slices',
                        arm_slices=slices, right_target=right)
    raise ValueError(f'No measured arm frame in {path}: {topic}')


def smooth_plan(start, target, duration, rate, max_velocity, max_acceleration):
    """Quintic blend: zero endpoint velocity/acceleration; duration lengthened to bound derivatives."""
    start, target = joints(start, 7), joints(target, 7)
    distance = max(abs(b-a) for a, b in zip(start, target))
    duration = max(duration, 1.875 * distance / max_velocity,
                   math.sqrt((10 / math.sqrt(3)) * distance / max_acceleration))
    count = max(1, math.ceil(duration * rate))
    return start, target, count, count / rate


def smooth_point(start, target, index, count):
    u = index / count
    blend = u*u*u * (10 + u * (-15 + 6*u))
    return [a + blend * (b-a) for a, b in zip(start, target)]


def move_right_arm(reference, config, read_arm, send_checked, check, wait, clock, emit,
                   previous_command=None):
    """Caller supplies fresh real feedback and checked 14-joint arm-only sender. Never commands fingers.

    No collision checking is claimed. Every scheduled sample is sent without catch-up bursts.
    Pauses stretch the trajectory instead of skipping to a later waypoint.
    """
    _, actual = read_arm()
    actual = joints(actual, 14)
    left = (joints(previous_command, 14) if previous_command is not None else actual)[:7]
    target = joints(reference['right_target'], 7)
    start, target, count, duration = smooth_plan(actual[7:], target, config['duration'],
        config['rate_hz'], config['max_velocity'], config['max_acceleration'])
    emit('right_arm_entry_started', reference=reference, left_hold=left, right_start=start,
         duration_seconds=duration, steps=count, max_velocity=config['max_velocity'],
         max_acceleration=config['max_acceleration'])

    def feedback():
        stamp, q = read_arm()
        q = joints(q, 14)
        if max(abs(a-b) for a, b in zip(q[:7], left)) > config['left_hold_tolerance']:
            raise RuntimeError('Left arm drifted during right-arm entry; refusing further commands')
        return stamp, q

    for index in range(count + 1):
        check()
        feedback()
        command = left + smooth_point(start, target, index, count)
        emit('right_arm_entry_command', index=index, command_rad=command)
        send_checked(command)  # Existing SafetySupervisor is mandatory in the Demo.
        wait(duration / count)  # No elapsed-time jumps if inference/OS/operator pauses this loop.
    deadline, stable_stamp, previous_stamp = clock() + config['arrival_timeout'], None, None
    next_report = float('-inf')
    while True:
        check()
        stamp, q = feedback()
        if not math.isfinite(stamp) or previous_stamp is not None and stamp < previous_stamp:
            raise RuntimeError('Feedback clock moved backwards during right-arm arrival check')
        previous_stamp = stamp
        residual = [desired - measured for measured, desired in zip(q[7:], target)]
        worst_index = max(range(7), key=lambda i: abs(residual[i]))
        error = abs(residual[worst_index])
        report = dict(right_target=target, right_measured=q[7:], error_rad=residual,
                      max_error_rad=error, worst_joint=f'zarm_r{worst_index + 1}_joint',
                      arrival_tolerance=config['arrival_tolerance'], feedback_stamp=stamp)
        if clock() >= next_report:
            emit('right_arm_entry_feedback', **report, remaining_seconds=max(0., deadline - clock()))
            next_report = clock() + .5
        stable_stamp = (stamp if stable_stamp is None else stable_stamp) if error <= config['arrival_tolerance'] else None
        if stable_stamp is not None and stamp - stable_stamp >= config['stable_seconds']:
            emit('right_arm_entry_arrived', max_error_rad=error, right_measured=q[7:])
            return
        if clock() >= deadline:
            emit('right_arm_entry_timeout', **report)
            raise TimeoutError(f'Right arm did not reach reference; joint={report["worst_joint"]}; '
                               f'target={target[worst_index]:.5f}; measured={q[7 + worst_index]:.5f}; '
                               f'max_error_rad={error:.4f}; tolerance={config["arrival_tolerance"]:.4f}')
        send_checked(left + target)
        wait(1 / config['rate_hz'])
