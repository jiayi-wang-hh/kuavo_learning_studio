"""Offline tests: no ROS, SDK, model or robot connection."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from kuavo_deploy.utils.recovery_entry import (measured_arm_reference, load_bag_right_reference,
                                              smooth_plan, smooth_point, move_right_arm)


class EntryTests(unittest.TestCase):
    @staticmethod
    def config():
        return dict(duration=3, rate_hz=100, max_velocity=.25, max_acceleration=.5,
                    arrival_tolerance=.03, left_hold_tolerance=.1, arrival_timeout=1,
                    stable_seconds=.5)

    def test_measured_radians_mapping_and_invalid_input(self):
        q=list(range(28))
        msg=SimpleNamespace(joint_data=SimpleNamespace(joint_q=q))
        self.assertEqual(measured_arm_reference(msg, [[12,19],[19,26]])[7:], list(range(19,26)))
        with self.assertRaises(ValueError):
            measured_arm_reference(msg, [[12,18],[19,26]])
        q[20]=float('nan')
        with self.assertRaises(ValueError):
            measured_arm_reference(msg, [[12,19],[19,26]])

    def test_bag_reads_first_measured_frame_and_checks_limits(self):
        q=[0.0]*28
        q[19:26]=[.1,.2,.3,.4,.5,.6,.7]
        msg=SimpleNamespace(_type='kuavo_msgs/sensorsData', joint_data=SimpleNamespace(joint_q=q))
        stamp=SimpleNamespace(to_sec=lambda: 123)
        class Bag:
            def __init__(self,*_): pass
            def __enter__(self): return self
            def __exit__(self,*_): pass
            def read_messages(self,topics):
                yield topics[0],msg,stamp
                raise AssertionError('Must stop after first frame')
        with patch.dict('sys.modules', rosbag=SimpleNamespace(Bag=Bag)):
            result=load_bag_right_reference('/tmp/example.bag','/sensors_data_raw',[[12,19],[19,26]],[-3.14]*14,[3.14]*14)
            self.assertEqual(result['right_target'],q[19:26])
            q[19]=10
            with self.assertRaises(ValueError):
                load_bag_right_reference('/tmp/example.bag','/sensors_data_raw',[[12,19],[19,26]],[-3.14]*14,[3.14]*14)

    def test_quintic_endpoints_and_speed_acceleration_bounds(self):
        a,b,n,d=smooth_plan([0]*7,[1.5,-.5,.2,0,0,0,0],3,100,.25,.5)
        self.assertGreater(d,3)
        points=[smooth_point(a,b,i,n) for i in range(n+1)]
        self.assertEqual(points[0],a)
        self.assertEqual(points[-1],b)
        dt=d/n
        speed=[[(y-x)/dt for x,y in zip(p,q)] for p,q in zip(points,points[1:])]
        self.assertLessEqual(max(abs(x) for v in speed for x in v),.25+1e-8)
        acceleration=[(y-x)/dt for p,q in zip(speed,speed[1:]) for x,y in zip(p,q)]
        self.assertLessEqual(max(abs(x) for x in acceleration),.5+1e-8)
        self.assertEqual(points[-1][3:],[0]*4)

    def fixture(self, follow=True):
        state=dict(time=0., q=[.05]*7+[0.]*7, commands=[], events=[])
        def wait(t): state['time']+=t
        def send(q):
            state['commands'].append(q)
            if follow: state['q']=q[:]
        return state,wait,send

    def run_motion(self,state,wait,send,check=lambda:None):
        return move_right_arm(dict(right_target=[.3]*7),self.config(),
            lambda:(state['time'],state['q']),send,check,wait,lambda:state['time'],
            lambda event,**fields:state['events'].append((event,fields)),
            previous_command=[.05]*7+[0.]*7)

    def test_only_right_arm_changes_and_arrival_requires_feedback(self):
        state,wait,send=self.fixture()
        self.run_motion(state,wait,send)
        self.assertTrue(all(q[:7]==[.05]*7 for q in state['commands']))
        self.assertTrue(all(len(q)==14 for q in state['commands']))  # no finger command
        self.assertEqual(state['commands'][-1][7:],[.3]*7)
        self.assertEqual(state['events'][-1][0],'right_arm_entry_arrived')
        state,wait,send=self.fixture(follow=False)
        with self.assertRaises(TimeoutError): self.run_motion(state,wait,send)
        self.assertNotIn('right_arm_entry_arrived',[e for e,_ in state['events']])

    def test_stop_and_safety_failures_propagate_without_further_commands(self):
        state,wait,send=self.fixture()
        def check():
            if len(state['commands'])==3: raise InterruptedError('stop')
        with self.assertRaises(InterruptedError): self.run_motion(state,wait,send,check)
        self.assertEqual(len(state['commands']),3)
        state,wait,send=self.fixture()
        def rejected(q): raise RuntimeError('SafetySupervisor rejected')
        with self.assertRaisesRegex(RuntimeError,'SafetySupervisor'): self.run_motion(state,wait,rejected)
        self.assertFalse(state['commands'])

    def test_timeout_reports_measured_joint_error_without_relaxing_tolerance(self):
        state,wait,send=self.fixture(follow=False)
        with self.assertRaisesRegex(TimeoutError, r'joint=zarm_r1_joint.*tolerance=0.0300'):
            self.run_motion(state,wait,send)
        reports=[fields for event,fields in state['events'] if event=='right_arm_entry_timeout']
        self.assertEqual(len(reports),1)
        self.assertEqual(reports[0]['right_measured'],[0.]*7)
        self.assertEqual(reports[0]['error_rad'],[.3]*7)
        self.assertNotIn('right_arm_entry_arrived',[event for event,_ in state['events']])

    def test_longer_arrival_wait_accepts_delayed_real_feedback(self):
        state,wait,_=self.fixture(follow=False)
        def delayed_send(q):
            state['commands'].append(q)
            if state['time'] >= 5: state['q']=q[:]
        config=self.config()
        config['arrival_timeout']=4
        move_right_arm(dict(right_target=[.3]*7),config,
            lambda:(state['time'],state['q']),delayed_send,lambda:None,wait,lambda:state['time'],
            lambda event,**fields:state['events'].append((event,fields)),
            previous_command=[.05]*7+[0.]*7)
        self.assertGreater(state['time'],5)
        self.assertEqual(state['events'][-1][0],'right_arm_entry_arrived')

    def test_pause_does_not_skip_waypoints_and_left_drift_aborts(self):
        state,wait,send=self.fixture()
        def check():
            if len(state['commands'])==3: state['time']+=20  # operator pause
        self.run_motion(state,wait,send,check)
        self.assertLessEqual(max(abs(q[7]-p[7]) for p,q in zip(state['commands'],state['commands'][1:])),.25/100+1e-8)
        state,wait,send=self.fixture()
        state['q'][0]=.5
        with self.assertRaisesRegex(RuntimeError,'Left arm drifted'): self.run_motion(state,wait,send)
        self.assertFalse(state['commands'])


if __name__=='__main__': unittest.main()
