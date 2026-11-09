"""User-run offline tests of the auto-demo gates. No ROS, weights or robot imports."""
import ast
from pathlib import Path
import unittest

from kuavo_deploy.utils.skill_demo import FreshVotes, expected_metadata


def judgment(label, stamp, *, accepted=True, status='accepted', epoch=0):
    return dict(event='judgment', status=status, accepted_current=accepted,
                parsed={'skill': label} if label else None,
                frame=dict(stamp_s=stamp, stamp_ns=int(stamp * 1e9), received_mono=stamp, epoch=epoch))


class VoteTests(unittest.TestCase):
    def test_post_stop_single_flip_waits_then_stable_normal_cancels(self):
        from kuavo_deploy.utils.skill_demo import PostStopDecision
        votes = FreshVotes()
        for t in [100, 101, 102]:
            votes.feed(judgment('recovery', t), t+.2, t+.2)
        votes.reset(not_before_ros=103)
        decision = PostStopDecision(deadline=115, normal_hits=3, recovery_hits=3)
        votes.feed(judgment('recovery', 102.5), 103, 103)
        self.assertIsNone(decision.evaluate(votes.snapshot(103, 103), 103))
        for t in [103, 104]:
            votes.feed(judgment('normal', t), t+.2, t+.2)
            self.assertIsNone(decision.evaluate(votes.snapshot(t+.2, t+.2), t+.2))
        votes.feed(judgment('normal', 105), 105.2, 105.2)
        self.assertEqual(decision.evaluate(votes.snapshot(105.2, 105.2), 105.2), 'normal')

    def test_post_stop_recovery_requires_three_new_fresh_votes_and_timeout(self):
        from kuavo_deploy.utils.skill_demo import PostStopDecision
        decision = PostStopDecision(deadline=110, normal_hits=3, recovery_hits=3)
        votes = FreshVotes()
        for t, skill in [(100, 'recovery'), (101, 'normal'), (102, 'recovery'), (103, 'recovery')]:
            votes.feed(judgment(skill, t), t+.2, t+.2)
            self.assertIsNone(decision.evaluate(votes.snapshot(t+.2, t+.2), t+.2))
        votes.feed(judgment('recovery', 104), 104.2, 104.2)
        self.assertEqual(decision.evaluate(votes.snapshot(104.2, 104.2), 104.2), 'recovery')
        self.assertIsNone(decision.evaluate(votes.snapshot(106, 106), 106))
        with self.assertRaises(TimeoutError):
            decision.evaluate(votes.snapshot(110, 110), 110)

    def test_recovery_initial_open_hand_never_completes_cycle(self):
        from kuavo_deploy.utils.skill_demo import RecoveryGripperCycle
        cycle = RecoveryGripperCycle(self.cycle_spec())
        for i in range(100):
            self.assertEqual(cycle.update(0.0, 0.01, i * .1), 'waiting_for_grasp')

    @staticmethod
    def cycle_spec():
        return dict(max_observation_age=.5, left_grasp_command_threshold=.5,
                    left_grasp_feedback_threshold=.25, grasp_stable_seconds=.3,
                    left_release_threshold=.15, release_stable_seconds=.5)

    def test_recovery_requires_measured_closure_then_stable_release(self):
        from kuavo_deploy.utils.skill_demo import RecoveryGripperCycle
        cycle = RecoveryGripperCycle(self.cycle_spec())
        for t in [0, .2, .4]:
            self.assertEqual(cycle.update(.9, .01, t), 'waiting_for_grasp')
        self.assertEqual(cycle.update(.9, .4, .5), 'waiting_for_grasp')
        self.assertEqual(cycle.update(.9, .4, .81), 'grasp_observed')
        self.assertEqual(cycle.update(0, .4, .9), 'grasp_observed')
        self.assertEqual(cycle.update(0, .01, 1), 'grasp_observed')
        self.assertEqual(cycle.update(0, .01, 1.3), 'grasp_observed')
        self.assertEqual(cycle.update(0, .01, 1.51), 'released')
        self.assertEqual(cycle.update(.8, .3, 1.6), 'grasp_observed')
        self.assertEqual(RecoveryGripperCycle(self.cycle_spec()).stage, 'waiting_for_grasp')

    def test_transient_closure_and_sample_gaps_do_not_complete_grasp(self):
        from kuavo_deploy.utils.skill_demo import RecoveryGripperCycle
        cycle = RecoveryGripperCycle(self.cycle_spec())
        cycle.update(.9, .4, 0)
        cycle.update(0, 0, .2)
        self.assertEqual(cycle.update(.9, .4, .31), 'waiting_for_grasp')
        self.assertEqual(cycle.update(.9, .4, 2), 'waiting_for_grasp')

    def test_exactly_three_different_fresh_recovery_frames(self):
        votes = FreshVotes()
        for i in range(3):
            votes.feed(judgment('recovery', 100 + i), 100.5 + i, 100.5 + i)
            s = votes.snapshot(100.5 + i, 100.5 + i)
            self.assertEqual(s['streak'], i + 1)
            self.assertTrue(s['fresh'])
        votes.feed(judgment('recovery', 102), 102.6, 102.6)
        self.assertEqual(votes.snapshot(102.6, 102.6)['streak'], 3)

    def test_normal_invalid_and_stale_results_break_recovery_sequence(self):
        votes = FreshVotes()
        votes.feed(judgment('recovery', 100), 100.2, 100.2)
        votes.feed(judgment('recovery', 101), 101.2, 101.2)
        votes.feed(judgment('normal', 102), 102.2, 102.2)
        votes.feed(judgment('recovery', 103), 103.2, 103.2)
        self.assertEqual(votes.snapshot(103.2, 103.2)['streak'], 1)
        votes.feed(judgment(None, 104, accepted=False, status='invalid_output'), 104.2, 104.2)
        self.assertEqual(votes.snapshot(104.2, 104.2)['streak'], 0)
        votes.feed(judgment('recovery', 105), 107, 107)
        self.assertFalse(votes.snapshot(107, 107)['fresh'])
        self.assertEqual(votes.snapshot(107, 107)['streak'], 0)

    def test_valid_vote_waits_can_have_short_display_gaps_but_not_long_gaps(self):
        votes = FreshVotes(max_age=1.5, max_gap=2.5)
        votes.feed(judgment('recovery', 100), 100.8, 100.8)
        self.assertFalse(votes.snapshot(101.6, 101.6)['fresh'])
        votes.feed(judgment('recovery', 101), 101.8, 101.8)
        self.assertEqual(votes.snapshot(101.8, 101.8)['streak'], 2)
        votes.feed(judgment('recovery', 104), 104.8, 104.8)
        self.assertEqual(votes.snapshot(104.8, 104.8)['streak'], 1)

    def test_switch_boundary_does_not_reuse_in_motion_votes(self):
        votes = FreshVotes()
        for t in [100,101,102]:
            votes.feed(judgment('normal', t), t+.5, t+.5)
        votes.reset(not_before_ros=103)
        votes.feed(judgment('normal', 102.5), 103, 103)
        self.assertEqual(votes.snapshot(103, 103)['streak'], 0)
        votes.feed(judgment('normal', 103), 103.5, 103.5)
        self.assertEqual(votes.snapshot(103.5, 103.5)['streak'], 1)

    def test_receipt_cannot_make_old_header_current(self):
        votes = FreshVotes()
        row = judgment('recovery', 100)
        row['frame']['received_mono'] = 200
        votes.feed(row, 200, 200)
        self.assertFalse(votes.snapshot(200, 200)['fresh'])

    def test_wrong_checkpoint_norm_pair_is_rejected(self):
        expected = dict(checkpoint='/tmp/normal', training_config_path='/tmp/normal.yaml', robot_norm_path='/tmp/normal.json')
        response = dict(status='ok', adapter='lingbot_vla', which_arm='both', execution_horizon=25, **expected)
        expected_metadata(response, expected)
        response['robot_norm_path'] = '/tmp/recovery.json'
        with self.assertRaises(RuntimeError):
            expected_metadata(response, expected)

    def test_monitor_resume_requires_fresh_stable_real_label(self):
        from kuavo_deploy.utils.skill_demo import stable_monitor_vote
        for skill in ['normal', 'recovery']:
            row = dict(last={'skill': skill}, fresh=True, streak=2)
            self.assertFalse(stable_monitor_vote(row, 3, 3))
            row['streak'] = 3
            self.assertTrue(stable_monitor_vote(row, 3, 3))
            row['fresh'] = False
            self.assertFalse(stable_monitor_vote(row, 3, 3))
        self.assertFalse(stable_monitor_vote(dict(last=None, fresh=False, streak=99), 3, 3))
        self.assertFalse(stable_monitor_vote(dict(last={'skill': 'invalid'}, fresh=True, streak=99), 3, 3))

    def test_preflight_has_no_top_level_robot_model_import(self):
        path = Path(__file__).resolve().parents[1] / 'src/scripts/script_skill_demo.py'
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module, ['rospy', 'torch', 'kuavo_deploy.kuavo_env',
                                              'kuavo_deploy.src.scripts.script'])


if __name__ == '__main__':
    unittest.main()
