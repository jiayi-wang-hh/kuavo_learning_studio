"""Read-only prefix observer, immutable observation association, and fail-open JSONL logging."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
import queue
import threading
import time
import uuid

import torch
import torch.nn.functional as F

from progress_monitor.extract_lingbot_features import pool_prefix
from progress_monitor.train_linear_probe import CLASSES

DEFAULT_PROBE = Path('/media/data/jiayi/outputs/lingbot_progress_pilot/full_probes/vision_unweighted/best.pt')
DEFAULT_LOG = Path('/media/data/jiayi/outputs/lingbot_progress_pilot/online_monitor/process.jsonl')
TIMING_KEY = '_progress_monitor_observation'
LOG = logging.getLogger(__name__)


class VisionProbe:
    """CPU head loaded without random initialization or RNG/state changes to the VLA."""
    def __init__(self, path=DEFAULT_PROBE):
        self.path = Path(path).expanduser().resolve(strict=True)
        c = torch.load(self.path, map_location='cpu', weights_only=True)
        if tuple(c['classes']) != CLASSES or c['config']['pooling'] != 'vision' or c['config']['weighting'] != 'unweighted':
            raise ValueError('Expected vision_unweighted with P1,P2,P2a,P2b,P3,P4 class order')
        self.mean, self.std = c['mean'].float(), c['std'].float()
        self.weight, self.bias = c['state_dict']['weight'].float(), c['state_dict']['bias'].float()
        for tensor, shape in ((self.mean, (2048,)), (self.std, (2048,)), (self.weight, (6, 2048)), (self.bias, (6,))):
            if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
                raise ValueError('Invalid probe tensor shape/value')
        if not (self.std > 0).all():
            raise ValueError('Invalid training standard deviation')
        self.provenance = c['provenance']
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    @torch.inference_mode()
    def probabilities(self, feature):
        x = torch.as_tensor(feature, dtype=torch.float32, device='cpu').reshape(1, 2048)
        if not torch.isfinite(x).all():
            raise ValueError('Nonfinite monitor feature')
        p = F.linear((x - self.mean) / self.std, self.weight, self.bias).softmax(-1)[0]
        if not torch.isfinite(p).all():
            raise ValueError('Nonfinite monitor probabilities')
        return p.tolist()


class PassiveProgressMonitor:
    def __init__(self, probe_path=DEFAULT_PROBE, log_path=DEFAULT_LOG, *, checkpoint, camera_order,
                 qwen_path, allow_input_mismatch=False, queue_size=8):
        self.enabled = False
        self.failure = None
        self._context = None
        self.allow_input_mismatch = allow_input_mismatch
        self.probe = VisionProbe(probe_path)
        expected = self.probe.provenance
        if Path(checkpoint).resolve() != Path(expected['checkpoint']).resolve():
            raise ValueError('VLA checkpoint differs from probe training checkpoint')
        if list(camera_order) != expected['camera_order'] or Path(qwen_path).resolve() != Path(expected['qwen']).resolve():
            raise ValueError('Camera order or Qwen processor differs from training')
        if expected['preprocessing'] != 'VLADataset Resize + FeatureTransform.apply':
            raise ValueError('Unknown offline preprocessing contract')
        self._queue = queue.Queue(maxsize=queue_size)
        self._closed = False
        self._stream = Path(log_path).expanduser()
        self._stream.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._stream.open('a', buffering=1)
        self.enabled = True
        self._write(dict(event='monitor_started', timestamp=time.time(), probe=str(self.probe.path),
                         probe_sha256=self.probe.sha256, classes=CLASSES,
                         feature='final_normalized_prefix.vision_masked_mean',
                         input_policy='allow_and_log_mismatch' if allow_input_mismatch else 'strict'))
        self._thread = threading.Thread(target=self._worker, name='passive-progress-monitor', daemon=True)
        self._thread.start()

    def _write(self, record):
        self._file.write(json.dumps(record, allow_nan=False) + '\n')
        self._file.flush()

    def disable(self, error):
        self.enabled = False
        self.failure = str(error)
        self._context = None
        LOG.warning('Passive progress monitor disabled (actions continue): %s', error)

    def begin(self, observation, *, preprocessing='PIL.Resize(BILINEAR,224) + FeatureTransform.apply'):
        """Snapshot only metadata from the exact request; never fetch a newer frame."""
        self._context = None
        if not self.enabled:
            return
        try:
            issues = []
            if preprocessing != self.probe.provenance['preprocessing']:
                issues.append('resize: online PIL versus offline torchvision tensor Resize')
            prompt = str(observation.get('prompt', ''))
            if prompt not in self.probe.provenance['task_instructions']:
                issues.append('task instruction differs from offline training')
            for camera in self.probe.provenance['camera_order']:
                if camera not in observation:
                    raise ValueError('missing camera: ' + camera)
            if issues and not self.allow_input_mismatch:
                raise ValueError('; '.join(issues))
            now = time.time()
            timing = deepcopy(observation.get(TIMING_KEY, {}))
            self._context = dict(observation_id=timing.get('observation_id') or uuid.uuid4().hex,
                                 server_received_timestamp=now, timing=timing,
                                 prompt=prompt, input_mismatches=issues,
                                 preprocessing=preprocessing, input_exact_match=not issues)
        except Exception as error:
            self.disable(error)

    def end(self):
        self._context = None

    def observe_prefix(self, hidden, prefix_pad_masks, image_masks, language_masks):
        if not self.enabled or self._context is None:
            return
        try:
            context = deepcopy(self._context)
            # Consume at most one prefix per inference request, never suffix steps.
            self._context = None
            if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != 2048:
                raise ValueError('Live monitor requires one observation and width 2048')
            vision_tokens = hidden.shape[1] - language_masks.shape[-1]
            cameras = image_masks.reshape(1, -1)
            if vision_tokens <= 0 or vision_tokens % cameras.shape[1]:
                raise ValueError('Unexpected prefix token layout')
            image_pad = cameras.bool().repeat_interleave(vision_tokens // cameras.shape[1], dim=1)
            mask = prefix_pad_masks.detach().clone().bool()
            mask[:, :vision_tokens] &= image_pad.to(mask.device)
            pooled = pool_prefix(hidden.detach(), mask, vision_tokens, 'vision')
            # Match offline NPZ float16 storage before the float32 probe normalization.
            feature = pooled.to(dtype=torch.float16, device='cpu').float().clone()
            self.submit_feature(feature, context)
        except Exception as error:
            self.disable(error)

    def submit_feature(self, feature, context):
        """Also supports CPU replay of cached offline features; never updates a head."""
        if self.enabled:
            self._queue.put_nowait((deepcopy(context), feature.detach().cpu().float().clone()))

    def _worker(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if not self.enabled:
                    continue
                context, feature = item
                probabilities = self.probe.probabilities(feature)
                now = time.time()
                timing = context['timing']
                snapshot = timing.get('snapshot_wall_timestamp')
                at_snapshot = timing.get('frame_age_at_snapshot_ms')
                elapsed = None if snapshot is None else 1000 * (now - float(snapshot))
                age = None if at_snapshot is None or elapsed is None or elapsed < 0 else float(at_snapshot) + elapsed
                index = max(range(6), key=probabilities.__getitem__)
                self._write(dict(event='prediction', timestamp=now, **context,
                                 frame_age_ms=age, frame_age_clock='client_ROS_age_plus_wall_elapsed' if age is not None else 'unknown',
                                 server_observation_age_ms=1000*(now-context['server_received_timestamp']),
                                 predicted_process=CLASSES[index], confidence=probabilities[index],
                                 probabilities=dict(zip(CLASSES, probabilities))))
            except Exception as error:
                self.disable(error)
            finally:
                self._queue.task_done()

    def close(self):
        if self._closed:
            return
        self._queue.join()
        self._queue.put(None)
        self._thread.join(timeout=5)
        self._file.close()
        self._closed = True
