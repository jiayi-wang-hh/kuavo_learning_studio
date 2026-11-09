"""Offline replay or manual inference-only A/B checks. No ROS, sockets, or robot actions."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch

from progress_monitor.online_monitor import DEFAULT_PROBE, PassiveProgressMonitor, VisionProbe, TIMING_KEY
from progress_monitor.extract_lingbot_features import CHECKPOINT, DATASET, LABELS, QWEN, LINGBOT, read_episode


def records(path):
    return [row for row in map(json.loads, path.read_text().splitlines()) if row['event'] == 'prediction']


def replay(args):
    probe = VisionProbe(args.probe)
    if args.features is None:
        raise ValueError('replay requires --features episode_NNN.npz')
    with np.load(args.features) as archive:
        meta = json.loads(str(archive['metadata_json']))
        for key in ('checkpoint', 'qwen', 'camera_order', 'preprocessing', 'task_instructions'):
            if meta[key] != probe.provenance[key]:
                raise ValueError('Replay feature provenance mismatch: ' + key)
        valid = archive['valid'] & archive['feature_valid'] & ~archive['overlap']
        indices = np.flatnonzero(valid)[:args.frames]
        if not len(indices):
            raise ValueError('No replay frames')
        features = archive['features_vision'][indices].copy()
        frame_ids = archive['frame_index'][indices].copy()
        ep = int(archive['episode_index'])
    log = args.output / 'replay.jsonl'
    monitor = PassiveProgressMonitor(args.probe, log, checkpoint=probe.provenance['checkpoint'],
                                     qwen_path=probe.provenance['qwen'], camera_order=probe.provenance['camera_order'])
    expected = []
    for feature, frame_id in zip(features, frame_ids):
        context = dict(observation_id=f'replay_episode_{ep:03d}_frame_{int(frame_id)}',
                       server_received_timestamp=time.time(), timing={},
                       prompt=probe.provenance['task_instructions'][0], input_mismatches=[],
                       preprocessing=probe.provenance['preprocessing'], input_exact_match=True,
                       replay=True, episode_index=ep, frame_index=int(frame_id))
        monitor.submit_feature(torch.from_numpy(feature), context)
        # The replay is deliberately bounded; drain between rows to avoid artificial overload.
        monitor._queue.join()
        expected.append(probe.probabilities(feature))
    monitor.close()
    output = records(log)
    if len(output) != len(indices) or monitor.failure:
        raise AssertionError(f'Replay monitor failed: {monitor.failure}')
    for row, expected_p, frame_id in zip(output, expected, frame_ids):
        np.testing.assert_allclose(list(row['probabilities'].values()), expected_p, atol=1e-7)
        assert row['frame_index'] == int(frame_id)
    reference = args.reference_predictions or args.probe.parent / 'validation_predictions.npz'
    reference_error = None
    if reference.is_file():
        with np.load(reference) as archive:
            lookup = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(archive['episode_index'], archive['frame_index']))}
            expected_rows = [lookup[(ep, int(f))] for f in frame_ids]
            saved_p = archive['probabilities'][expected_rows]
            actual_p = np.asarray(expected)
            # CPU/GPU float32 GEMM can differ slightly; class association must still match.
            np.testing.assert_allclose(actual_p, saved_p, atol=1e-4, rtol=1e-4)
            np.testing.assert_array_equal(actual_p.argmax(1), archive['predicted'][expected_rows])
            reference_error = float(np.max(np.abs(actual_p - saved_p)))
    return dict(mode='replay', frames=len(indices), probability_parity=True, observation_binding=True,
                offline_reference_checked=reference_error is not None, max_reference_probability_error=reference_error,
                probe=str(probe.path), probe_sha256=probe.sha256, log=str(log))


def inference_only(args):
    # This branch is a manual GPU job; the replay/help paths never import/load the VLA.
    import sys
    import yaml
    sys.path.insert(0, str(LINGBOT))
    from lerobot.datasets.video_utils import decode_video_frames
    from kuavo_server.adapters.lingbot_vla import LingBotVlaAdapter

    labels, rows, meta, info, tasks, selected = read_episode(args.dataset, args.labels, args.episode)
    adapter = LingBotVlaAdapter(checkpoint=str(args.checkpoint), model_repo_root=str(LINGBOT), which_arm='both',
                               execution_horizon=25, qwen25_path=str(args.qwen),
                               training_config_path=str(args.checkpoint / 'lingbotvla_cli.yaml'),
                               robot_config_path=str(LINGBOT / 'configs/robot_configs/kuavo.yaml'),
                               robot_norm_path=str(LINGBOT / 'assets/norm_stats/kuavo_package.json'),
                               num_denoising_step=10, use_compile=False, use_fp32=False, progress_monitor=True,
                               progress_monitor_checkpoint=str(args.probe), progress_monitor_log=str(args.output / 'inference.jsonl'),
                               progress_monitor_allow_input_mismatch=args.allow_input_mismatch)
    monitor = adapter._progress_monitor
    if monitor is None:
        raise RuntimeError(adapter._progress_monitor_error)
    prompt = yaml.safe_load(args.deploy_config.read_text())['inference']['task_prompt']
    cases = []
    try:
        for i in selected[:args.frames]:
            observation = {'observation.state': rows['observation.state'][i].astype(np.float32), 'prompt': prompt}
            for camera in monitor.probe.provenance['camera_order']:
                prefix = 'videos/' + camera
                video = args.dataset / info['video_path'].format(video_key=camera, chunk_index=meta[prefix + '/chunk_index'],
                                                                file_index=meta[prefix + '/file_index'])
                stamp = float(rows['timestamp'][i]) + meta[prefix + '/from_timestamp']
                image = decode_video_frames(video, [stamp], 1 / info['fps'] - 1e-4, backend='torchcodec')[0]
                observation[camera] = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            observation_id = f'inference_episode_{args.episode:03d}_frame_{int(rows["frame_index"][i])}'
            # Historical video timestamps are not live wall-clock capture times.
            observation[TIMING_KEY] = dict(observation_id=observation_id, replay=True,
                                          episode_index=args.episode, frame_index=int(rows['frame_index'][i]))
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            adapter._progress_monitor = None
            adapter.model.vla.model._progress_prefix_observer = None
            baseline = adapter._infer_chunk(deepcopy(observation))
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            adapter._progress_monitor = monitor
            adapter.model.vla.model._progress_prefix_observer = monitor.observe_prefix
            observed = adapter._infer_chunk(deepcopy(observation))
            np.testing.assert_array_equal(baseline, observed)
            monitor._queue.join()
            if not monitor.enabled:
                raise AssertionError('Monitor disabled: ' + str(monitor.failure))
            cases.append(dict(observation_id=observation_id, actions_identical=True))
        if not cases:
            raise ValueError('No inference check frames selected')
    finally:
        monitor.close()
    predictions = records(args.output / 'inference.jsonl')
    assert [r['observation_id'] for r in predictions] == [r['observation_id'] for r in cases]
    return dict(mode='inference-only', cases=cases, predictions=len(predictions),
                actions_never_executed=True, input_mismatches=predictions[0]['input_mismatches'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('replay', 'inference-only'), required=True)
    parser.add_argument('--probe', type=Path, default=DEFAULT_PROBE)
    parser.add_argument('--features', type=Path, help='Cached episode NPZ for replay')
    parser.add_argument('--reference-predictions', type=Path, help='Offline predictions NPZ; default best.pt sibling validation_predictions.npz if present')
    parser.add_argument('--output', type=Path, required=True, help='New check output directory')
    parser.add_argument('--frames', type=int, default=1)
    parser.add_argument('--episode', type=int, default=10)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--labels', type=Path, default=LABELS)
    parser.add_argument('--qwen', type=Path, default=QWEN)
    parser.add_argument('--deploy-config', type=Path, default=Path('configs/deploy/deploy_lingbot_package.yaml'))
    parser.add_argument('--allow-input-mismatch', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.frames <= 20:
        parser.error('Use 1–20 frames for bounded checks')
    args.output.mkdir(parents=True, exist_ok=False)
    report = replay(args) if args.mode == 'replay' else inference_only(args)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
