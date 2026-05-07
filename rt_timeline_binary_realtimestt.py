"""
Real-time Binary Speaker Diarization (Enroll → Classify)

Workflow:
  Step 1. Enroll Speaker 1 and Speaker 2 — either by recording from an audio
          device until the user presses Stop, or by loading a pre-recorded
          audio file.
  Step 2. Run live diarization. Every speech window is classified as exactly
          one of the two enrolled speakers (pure binary classification, by
          cosine similarity to the two stored mean embeddings).  When live
          diarization is ended, the captured source audio is split into two
          time-aligned WAV streams using the diarized segment times.  During
          live diarization, the same time-aligned split is also fed into two
          RealtimeSTT recorders so each enrolled speaker gets independent STT.

No "Pending" layer, no online embedding updates, no auto-clustering — the two
mean embeddings are frozen at the moment enrollment finalises, and stay frozen
for the rest of the session.
"""

import soundcard as sc
import numpy as np
import time
import sys
import queue
import os
import wave
import threading
import logging
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton,
    QHBoxLayout, QScrollArea, QMessageBox, QLabel,
    QComboBox, QGroupBox, QFileDialog, QPlainTextEdit,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen, QFont

import torch
import torchaudio


# =====================================================================
# Configuration
# =====================================================================
NUM_SPEAKERS = 2
SPEAKER_COLORS = ["#FF4444", "#4488FF"]   # Speaker 1 = red, Speaker 2 = blue
SPEAKER_NAMES = ["Speaker 1", "Speaker 2"]

TIMELINE_HEIGHT = 300
TIMELINE_UPDATE_INTERVAL = 0.3
SIZE_UPDATE_INTERVAL = 1.0

SAMPLE_RATE = 16000
CHUNK_SIZE = 2048
WINDOW_SIZE = 1.0
WINDOW_PROCESS_INTERVAL = 0.1
WIN_SAMPLES = int(SAMPLE_RATE * WINDOW_SIZE)

DEVICE_PREF = "cuda"
VAD_THRESH = 0.5

# How long to wait for in-flight VAD/embedding tasks to drain after a file
# enrollment finishes streaming (max time, but we poll and finalise as soon
# as the queues empty out).
FILE_DRAIN_POLL_MS = 50
FILE_DRAIN_MAX_MS = 3000

# When the user ends live diarization, capture is paused first and the
# already-queued VAD/embedding work is allowed to drain before exporting WAVs.
DIARIZATION_DRAIN_POLL_MS = 50
DIARIZATION_DRAIN_MIN_MS = 200
DIARIZATION_DRAIN_MAX_MS = 5000

# Worker-thread WAV export settings.  Writing in blocks keeps memory bounded
# and keeps the Qt GUI event loop responsive during long exports.
EXPORT_BLOCK_SAMPLES = SAMPLE_RATE * 4
EXPORT_STATUS_INTERVAL_SEC = 0.5

# RealtimeSTT integration. Audio fed into RealtimeSTT must be 16 kHz mono
# raw 16-bit PCM. RealtimeSTT's current default buffer_size is 512 samples;
# using 1024 can make its internal Silero VAD raise "Provided number of
# samples is 1024". The splitter therefore emits continuous, time-aligned
# 512-sample chunks per speaker, with non-speaker regions filled with silence.
REALTIME_STT_ENABLED = True
STT_MODEL = "base"
STT_REALTIME_MODEL = "tiny"
STT_LANGUAGE = "ko"
STT_DEVICE_PREF = DEVICE_PREF
STT_COMPUTE_TYPE = "default"
STT_FEED_SAMPLES = 512
STT_SPLIT_COMMIT_DELAY_SEC = 1.8
STT_FEED_QUEUE_MAX = 400
STT_FINAL_SILENCE_SEC = 1.0


# Processor phases
PHASE_IDLE = "idle"
PHASE_ENROLL_S1 = "enroll_s1"
PHASE_ENROLL_S2 = "enroll_s2"
PHASE_DIARIZE = "diarize"


# =====================================================================
# Silero VAD (unchanged)
# =====================================================================
class SileroVAD(QThread):
    model_loaded = pyqtSignal()

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.vad_model = None
        self.get_speech_ts = None
        self.model_loaded_flag = False
        self.vad_queue = queue.Queue()
        self.res_queue = queue.Queue()
        self._stop_proc = False

    def run(self):
        try:
            print("Loading Silero VAD model on CPU...")
            model, utils = torch.hub.load(
                'snakers4/silero-vad', 'silero_vad',
                force_reload=False, onnx=False
            )
            self.vad_model = model.to("cpu")
            self.get_speech_ts = utils[0]
            self.model_loaded_flag = True
            print("Silero VAD model loaded.")
        except Exception as e:
            print(f"Error loading Silero VAD model: {e}")
            raise

        self.model_loaded.emit()

        while not self._stop_proc:
            try:
                task_id, audio_data, sr = self.vad_queue.get(timeout=0.1)
                is_speech = self._detect_speech(audio_data, sr)
                self.res_queue.put((task_id, is_speech))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"VAD processing error: {e}")

    def _detect_speech(self, audio_data, sr=16000):
        if not self.model_loaded_flag or self.vad_model is None or len(audio_data) < 1600:
            return False
        try:
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
            with torch.no_grad():
                ts = self.get_speech_ts(
                    audio_tensor, self.vad_model,
                    threshold=self.threshold,
                    sampling_rate=sr, return_seconds=False,
                )
            return len(ts) > 0
        except Exception as e:
            print(f"Speech detection error: {e}")
            return False

    def detect_async(self, audio_data, sr=16000):
        if not self.model_loaded_flag:
            return None
        task_id = time.time()
        self.vad_queue.put((task_id, audio_data.copy(), sr))
        return task_id

    def get_result(self):
        try:
            return self.res_queue.get_nowait()
        except queue.Empty:
            return None

    def stop_processing(self):
        self._stop_proc = True


# =====================================================================
# WeSpeaker embedding (unchanged)
# =====================================================================
class WeSpeakerEncoder(QThread):
    model_loaded = pyqtSignal()

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.model = None
        self.inference = None
        self.model_loaded_flag = False
        self.emb_queue = queue.Queue()
        self.res_queue = queue.Queue()
        self._stop_proc = False

    def run(self):
        try:
            print(f"Loading WeSpeaker model on {self.device.upper()}...")
            from pyannote.audio import Model, Inference
            self.model = Model.from_pretrained(
                "pyannote/wespeaker-voxceleb-resnet34-LM",
                use_auth_token=False,
            )
            self.model = self.model.to(torch.device(self.device))
            self.inference = Inference(self.model, window="whole")
            self.model_loaded_flag = True
            print("WeSpeaker model loaded.")
        except Exception as e:
            print(f"Error loading WeSpeaker model: {e}")
            raise

        self.model_loaded.emit()

        while not self._stop_proc:
            try:
                task_id, audio, sr = self.emb_queue.get(timeout=0.1)
                emb = self._compute_emb(audio, sr)
                self.res_queue.put((task_id, emb))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Embedding error: {e}")

    def _compute_emb(self, audio, sr=16000):
        if not self.model_loaded_flag or self.inference is None:
            return None
        try:
            waveform = torch.tensor(audio, dtype=torch.float32)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            audio_dict = {"waveform": waveform, "sample_rate": sr}
            with torch.no_grad():
                emb = self.inference(audio_dict)
            if isinstance(emb, torch.Tensor):
                return emb.cpu().numpy()
            return np.array(emb)
        except Exception as e:
            print(f"Embedding computation error: {e}")
            return None

    def embed_async(self, audio, sr=16000):
        if not self.model_loaded_flag:
            return None
        task_id = time.time()
        self.emb_queue.put((task_id, audio.copy(), sr))
        return task_id

    def get_res(self):
        try:
            return self.res_queue.get_nowait()
        except queue.Empty:
            return None

    def stop_proc(self):
        self._stop_proc = True


# =====================================================================
# Speaker handler — pure binary classification, no pending, no updates
# =====================================================================
class SpeakerHandler:
    """
    Stores up to NUM_SPEAKERS = 2 mean embeddings, computed once from
    enrollment audio and never updated thereafter. classify() returns the
    speaker id of whichever mean embedding has higher cosine similarity.
    """

    def __init__(self):
        self.num_speakers = NUM_SPEAKERS
        self.mean_embs = [None, None]
        self.enroll_embs = [[], []]
        self.curr_spk = None

    # ---- enrollment ----
    def add_enrollment_embedding(self, speaker_id, emb):
        if 0 <= speaker_id < self.num_speakers:
            self.enroll_embs[speaker_id].append(emb)

    def finalize_enrollment(self, speaker_id):
        if not self.enroll_embs[speaker_id]:
            return False
        embs = np.array(self.enroll_embs[speaker_id])
        # Use median for robustness against outliers (e.g. silence near
        # speech boundary that still passed VAD).
        self.mean_embs[speaker_id] = np.median(embs, axis=0)
        return True

    def enrollment_count(self, speaker_id):
        return len(self.enroll_embs[speaker_id])

    def is_enrolled(self, speaker_id):
        return self.mean_embs[speaker_id] is not None

    def both_enrolled(self):
        return all(m is not None for m in self.mean_embs)

    # ---- classification (binary, frozen embeddings) ----
    def classify(self, emb):
        if not self.both_enrolled():
            return None, 0.0
        emb_norm = emb / (np.linalg.norm(emb) + 1e-12)
        sims = []
        for mean_emb in self.mean_embs:
            mean_norm = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)
            sims.append(float(np.dot(mean_norm, emb_norm)))
        spk_id = int(np.argmax(sims))
        self.curr_spk = spk_id
        return spk_id, sims[spk_id]

    # ---- reset ----
    def reset_speaker(self, speaker_id):
        self.enroll_embs[speaker_id] = []
        self.mean_embs[speaker_id] = None

    def reset_all(self):
        self.curr_spk = None
        self.mean_embs = [None, None]
        self.enroll_embs = [[], []]


# =====================================================================
# Audio capture from device
# =====================================================================
class AudioCapture(QThread):
    chunk_ready = pyqtSignal(np.ndarray)

    def __init__(self, device_name=None, use_mic=False):
        super().__init__()
        self._running = True
        self._paused = True   # always start paused; resume() to begin
        self.use_mic = use_mic
        self.device_name = device_name
        self.device = None
        self._setup_device()

    def _setup_device(self):
        try:
            if self.device_name:
                self.device = sc.get_microphone(
                    id=self.device_name, include_loopback=not self.use_mic
                )
            else:
                if self.use_mic:
                    self.device = sc.default_microphone()
                else:
                    self.device = sc.get_microphone(
                        id=str(sc.default_speaker().name), include_loopback=True
                    )
        except Exception as e:
            print(f"Error setting up audio device: {e}")
            try:
                if self.use_mic:
                    self.device = sc.default_microphone()
                else:
                    self.device = sc.get_microphone(
                        id=str(sc.default_speaker().name), include_loopback=True
                    )
            except Exception as e2:
                print(f"Fallback audio device also failed: {e2}")
                self.device = None

    def run(self):
        if not self.device:
            print("No audio device available.")
            return
        # Re-open the recorder context whenever we resume so OS-buffered
        # samples accumulated while paused are dropped.
        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue
            try:
                with self.device.recorder(
                    samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE
                ) as recorder:
                    while self._running and not self._paused:
                        try:
                            audio_data = recorder.record(numframes=CHUNK_SIZE)
                            if audio_data.size == 0:
                                continue
                            if len(audio_data.shape) > 1:
                                audio_data = audio_data[:, 0]
                            audio_data = audio_data.flatten().astype(np.float32)
                            mx = np.max(np.abs(audio_data))
                            if mx > 1.0:
                                audio_data = audio_data / mx
                            self.chunk_ready.emit(audio_data)
                        except Exception as e:
                            print(f"Audio recording error: {e}")
                            time.sleep(0.1)
            except Exception as e:
                print(f"Error initialising audio recorder: {e}")
                time.sleep(0.5)

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._running = False


# =====================================================================
# Audio file source — for enrolling from a pre-recorded file
# =====================================================================
class AudioFileSource(QThread):
    chunk_ready = pyqtSignal(np.ndarray)
    finished_playing = pyqtSignal()

    def __init__(self, file_path):
        super().__init__()
        self._running = True
        self.file_path = file_path

    def run(self):
        try:
            print(f"Loading audio file: {self.file_path}")
            waveform, sr = torchaudio.load(self.file_path)

            # Mix to mono.
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample to SAMPLE_RATE if needed.
            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                waveform = resampler(waveform)

            audio = waveform.squeeze(0).numpy().astype(np.float32)

            mx = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
            if mx > 1.0:
                audio = audio / mx

            print(f"Audio loaded: {len(audio) / SAMPLE_RATE:.2f}s, "
                  f"{len(audio)} samples @ {SAMPLE_RATE} Hz")

            # Stream chunks at real-time pace so the VAD/embedding pipeline
            # doesn't get overwhelmed.
            num_samples = len(audio)
            idx = 0
            chunk_dur = CHUNK_SIZE / SAMPLE_RATE
            while self._running and idx < num_samples:
                end = min(idx + CHUNK_SIZE, num_samples)
                chunk = audio[idx:end]
                if len(chunk) < CHUNK_SIZE:
                    chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))
                self.chunk_ready.emit(chunk)
                idx = end
                time.sleep(chunk_dur)

            print("File playback finished.")
        except Exception as e:
            print(f"Audio file source error: {e}")
        finally:
            self.finished_playing.emit()

    def stop(self):
        self._running = False


# =====================================================================
# Segment / Timeline (no pending segments)
# =====================================================================
class Segment:
    def __init__(self, start_time, duration, spk_id=None, emb=None, is_speech=False):
        self.start_time = start_time
        self.duration = duration
        self.spk_id = spk_id      # always int 0/1, or None for non-speech
        self.emb = emb
        self.is_speech = is_speech

    @property
    def end_time(self):
        return self.start_time + self.duration


class Timeline:
    def __init__(self):
        self.segs = []
        self.active = False

    def start_timeline(self):
        self.active = True

    def pause_timeline(self):
        self.active = False

    def resume_timeline(self):
        self.active = True

    def add_seg(self, seg):
        if self.active:
            self.segs.append(seg)

    def get_segs(self):
        return self.segs

    def reset(self):
        self.segs = []


# =====================================================================
# Audio processor — phase-aware, queue-based
# =====================================================================
class AudioProcessor(QThread):
    tl_updated = pyqtSignal(list)
    enrollment_progress = pyqtSignal(int, int)   # (speaker_id, count)

    def __init__(self, encoder, vad_processor, spk_handler, tl_manager):
        super().__init__()
        self._running = True
        self._paused = True
        self.encoder = encoder
        self.vad_processor = vad_processor
        self.spk_handler = spk_handler
        self.tl_manager = tl_manager

        self.phase = PHASE_IDLE

        self.buffer = np.zeros(WIN_SAMPLES, dtype=np.float32)
        self.buf_idx = 0
        self.buf_full = False

        self.last_proc_time = 0
        self.pending_vad_tasks = {}
        self.pending_segs = {}
        self.force_process_window = False

        self.last_ui_update = 0
        self.pending_ui_update = False

        # Sample-based audio clock — resets on every phase change.
        self.total_samples = 0

        # Chunks arrive on whichever thread emits chunk_ready (queued
        # connection from AudioCapture/AudioFileSource → main thread).
        # We queue them and process in run() so processing is decoupled
        # from chunk arrival, and so pending VAD/embedding results keep
        # draining even after the source stops emitting.
        self.chunk_queue = queue.Queue(maxsize=2000)

        # Raw source audio captured during PHASE_DIARIZE.  This is the exact
        # mono stream that feeds the diarization clock, so exported speaker
        # WAVs can be generated by converting segment times → sample indices.
        self.diarize_audio_chunks = []
        self.diarize_total_samples = 0
        self.stt_splitter = None

    def set_stt_splitter(self, splitter):
        self.stt_splitter = splitter

    # ---- phase control ----
    def set_phase(self, phase):
        """Switch processing phase. Drops all in-flight state so the new
        phase starts with a clean slate (clean buffer, zeroed clock, no
        cross-phase VAD/embedding contamination)."""
        self.phase = phase
        self.buffer.fill(0)
        self.buf_idx = 0
        self.buf_full = False
        self.total_samples = 0
        self.pending_vad_tasks.clear()
        self.pending_segs.clear()
        self.last_proc_time = 0
        self.pending_ui_update = False
        self.force_process_window = False

        if phase == PHASE_DIARIZE:
            self.reset_diarization_audio()
            if self.stt_splitter is not None:
                self.stt_splitter.reset()
                self.stt_splitter.set_active(True)
        elif phase == PHASE_IDLE:
            # IDLE is used after export/reset/enrollment cleanup.  Keeping old
            # source audio here would make a later export ambiguous.
            self.reset_diarization_audio()
            if self.stt_splitter is not None:
                self.stt_splitter.set_active(False)
                self.stt_splitter.reset()

        # Drain any chunks that were queued before this transition.
        while True:
            try:
                self.chunk_queue.get_nowait()
            except queue.Empty:
                break
        # Drain stale VAD / embedding results.
        while self.vad_processor.get_result() is not None:
            pass
        while self.encoder.get_res() is not None:
            pass

    # ---- chunk intake (called from any thread) ----
    def add_chunk(self, audio_data):
        try:
            self.chunk_queue.put_nowait(audio_data)
        except queue.Full:
            # Drop chunk if queue is overwhelmed — better than blocking
            # the audio thread.
            pass

    # ---- main processing loop ----
    def run(self):
        while self._running:
            try:
                audio_data = self.chunk_queue.get(timeout=0.05)
                if not self._paused and self.phase != PHASE_IDLE:
                    audio_data = np.asarray(audio_data, dtype=np.float32).flatten()
                    if self.phase == PHASE_DIARIZE:
                        self._record_diarize_audio(audio_data)
                    self._add_buf(audio_data)
                    self.total_samples += len(audio_data)
                    now = time.time()
                    if now - self.last_proc_time >= WINDOW_PROCESS_INTERVAL:
                        self.last_proc_time = now
                        self._proc_window()
            except queue.Empty:
                pass

            # Always drain VAD / embedding results, even when paused, so
            # that pipeline tail emptying after an enrollment file ends
            # still completes.
            if (
                self.force_process_window
                and not self._paused
                and self.phase == PHASE_DIARIZE
                and self.chunk_queue.empty()
            ):
                self.force_process_window = False
                self._proc_window()

            self._proc_vad_results()
            self._proc_emb_res()
            self._check_ui_update()

    # ---- internals ----
    def _check_ui_update(self):
        now = time.time()
        if self.pending_ui_update and now - self.last_ui_update >= TIMELINE_UPDATE_INTERVAL:
            self.last_ui_update = now
            self.pending_ui_update = False
            self.tl_updated.emit(self.tl_manager.get_segs())

    def _record_diarize_audio(self, audio_chunk):
        chunk = np.asarray(audio_chunk, dtype=np.float32).flatten().copy()
        if chunk.size == 0:
            return
        self.diarize_audio_chunks.append(chunk)
        self.diarize_total_samples += len(chunk)
        if self.stt_splitter is not None:
            self.stt_splitter.add_audio(chunk)

    def reset_diarization_audio(self):
        self.diarize_audio_chunks = []
        self.diarize_total_samples = 0

    def get_diarization_audio_snapshot(self):
        if not self.diarize_audio_chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.diarize_audio_chunks).astype(np.float32, copy=False)

    def get_diarization_audio_chunks_snapshot(self):
        """Return a lightweight snapshot of captured diarization chunks.

        This intentionally does *not* concatenate/copy the whole recording on
        the GUI thread.  The app pauses capture and processing before export,
        then keeps the chunks alive until the background export worker finishes.
        """
        return list(self.diarize_audio_chunks), int(self.diarize_total_samples)

    def _add_buf(self, audio_chunk):
        n = len(audio_chunk)
        if self.buf_idx + n <= WIN_SAMPLES:
            self.buffer[self.buf_idx:self.buf_idx + n] = audio_chunk
            self.buf_idx += n
        else:
            rem = WIN_SAMPLES - self.buf_idx
            self.buffer[self.buf_idx:] = audio_chunk[:rem]
            self.buffer[:n - rem] = audio_chunk[rem:]
            self.buf_idx = n - rem
            self.buf_full = True

    def _get_window(self):
        if not self.buf_full:
            return self.buffer[:self.buf_idx] if self.buf_idx > 0 else None
        win = np.empty(WIN_SAMPLES, dtype=np.float32)
        win[:WIN_SAMPLES - self.buf_idx] = self.buffer[self.buf_idx:]
        win[WIN_SAMPLES - self.buf_idx:] = self.buffer[:self.buf_idx]
        return win

    def _proc_window(self):
        win = self._get_window()
        if win is None or len(win) < SAMPLE_RATE * 0.5:
            return
        actual_dur = len(win) / SAMPLE_RATE
        audio_t_now = self.total_samples / SAMPLE_RATE
        seg_start = audio_t_now - actual_dur
        vad_id = self.vad_processor.detect_async(win)
        if vad_id is not None:
            seg = Segment(seg_start, actual_dur, is_speech=False)
            self.pending_vad_tasks[vad_id] = (seg, seg_start, win.copy())

    def _proc_vad_results(self):
        while True:
            res = self.vad_processor.get_result()
            if res is None:
                break
            task_id, is_speech = res
            if task_id not in self.pending_vad_tasks:
                continue   # stale task from a previous phase
            seg, seg_time, win = self.pending_vad_tasks.pop(task_id)
            seg.is_speech = is_speech
            if is_speech:
                emb_id = self.encoder.embed_async(win)
                if emb_id is not None:
                    self.pending_segs[emb_id] = (seg, seg_time)
            else:
                # Show non-speech segments only during diarization.
                if self.phase == PHASE_DIARIZE:
                    self.tl_manager.add_seg(seg)
                    self._submit_segment_to_stt(seg)
                    self.pending_ui_update = True

    def _proc_emb_res(self):
        while True:
            res = self.encoder.get_res()
            if res is None:
                break
            task_id, emb = res
            if task_id not in self.pending_segs:
                continue
            seg, seg_time = self.pending_segs.pop(task_id)
            if emb is None:
                continue

            if self.phase == PHASE_ENROLL_S1:
                self.spk_handler.add_enrollment_embedding(0, emb)
                self.enrollment_progress.emit(0, self.spk_handler.enrollment_count(0))
            elif self.phase == PHASE_ENROLL_S2:
                self.spk_handler.add_enrollment_embedding(1, emb)
                self.enrollment_progress.emit(1, self.spk_handler.enrollment_count(1))
            elif self.phase == PHASE_DIARIZE:
                spk_id, _sim = self.spk_handler.classify(emb)
                seg.spk_id = spk_id
                seg.emb = emb
                self.tl_manager.add_seg(seg)
                self._submit_segment_to_stt(seg)
                self.pending_ui_update = True
            # If phase has changed to IDLE between queueing and result, the
            # embedding is silently discarded. That's the desired behaviour.

    def _submit_segment_to_stt(self, seg):
        if self.stt_splitter is not None and self.phase == PHASE_DIARIZE:
            self.stt_splitter.add_segment(seg)

    def flush_realtime_stt(self):
        if self.stt_splitter is not None:
            self.stt_splitter.flush(final=True)

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def request_final_window(self):
        # Queue one last classification window after capture stops so the tail
        # of the recorded source stream is represented in the timeline/export.
        self.force_process_window = True

    def stop(self):
        self._running = False

    # ---- helper for the GUI to know when pipeline has drained ----
    def pipeline_idle(self):
        return (
            self.chunk_queue.empty()
            and not self.force_process_window
            and not self.pending_vad_tasks
            and not self.pending_segs
        )


# =====================================================================
# Timeline UI — only 2 speakers + non-speech, no pending row
# =====================================================================
class TimelineUI(QWidget):
    def __init__(self):
        super().__init__()
        self.segs = []
        self.num_speakers = NUM_SPEAKERS
        self.pixels_per_second = 100
        self.setMinimumHeight(TIMELINE_HEIGHT)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor("#1A1A1A"))
        self.setPalette(pal)
        self.last_size_update = 0
        self.current_width = 800

    def update_segs(self, segs):
        self.segs = segs
        self._update_size_throttled()
        self.update()

    def _update_size_throttled(self):
        now = time.time()
        if now - self.last_size_update >= SIZE_UPDATE_INTERVAL:
            self.last_size_update = now
            self._update_size()

    def _update_size(self):
        if not self.segs:
            new_w = 800
        else:
            max_t = max(s.end_time for s in self.segs)
            new_w = max(800, int((max_t + 5) * self.pixels_per_second))
        if abs(new_w - self.current_width) > 50:
            self.current_width = new_w
            self.setMinimumWidth(new_w)

    @staticmethod
    def _format_time(seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QBrush(QColor("#1A1A1A")))

        total_layers = self.num_speakers + 1   # 2 speakers + non-speech
        layer_h = max(60, h // total_layers)

        max_t = max((s.end_time for s in self.segs), default=0)

        # Layer dividers
        painter.setPen(QPen(QColor("#444444"), 1))
        for i in range(1, total_layers):
            y = int(i * layer_h)
            painter.drawLine(0, y, w, y)

        # Time grid (every 10s)
        painter.setPen(QPen(QColor("#666666")))
        painter.setFont(QFont("Arial", 10))
        for i in range(0, int(max_t) + 10, 10):
            x = int(i * self.pixels_per_second)
            if x <= w:
                painter.drawLine(x, 0, x, h)
                painter.drawText(x + 5, 15, self._format_time(i))

        # Segments
        if self.segs:
            v_start = max(0, event.rect().left() / self.pixels_per_second - 1)
            v_end = (event.rect().right() / self.pixels_per_second) + 1
            for seg in self.segs:
                if seg.end_time < v_start or seg.start_time > v_end:
                    continue
                x0 = int(seg.start_time * self.pixels_per_second)
                x1 = int(seg.end_time * self.pixels_per_second)
                if x0 > w or x1 < 0:
                    continue
                x0 = max(0, x0)
                x1 = min(w, x1)
                if seg.is_speech and isinstance(seg.spk_id, int) and 0 <= seg.spk_id < self.num_speakers:
                    layer = seg.spk_id
                    color = QColor(SPEAKER_COLORS[seg.spk_id])
                else:
                    layer = self.num_speakers
                    color = QColor("#666666")
                color.setAlpha(150)
                y = int(layer * layer_h + 5)
                painter.fillRect(x0, y, x1 - x0, layer_h - 10, QBrush(color))

        # Layer labels
        painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        for i in range(self.num_speakers):
            painter.setPen(QPen(QColor(SPEAKER_COLORS[i])))
            y = int(i * layer_h + layer_h // 2 + 5)
            painter.drawText(10, y, SPEAKER_NAMES[i])
        painter.setPen(QPen(QColor("#AAAAAA")))
        y = int(self.num_speakers * layer_h + layer_h // 2 + 5)
        painter.drawText(10, y, "Non-Speech")


# =====================================================================
# Export helpers — segment times → source-audio sample ranges → 2 WAV stems
# =====================================================================
def split_audio_by_timeline(audio, segs, sample_rate=SAMPLE_RATE):
    """Return two time-aligned mono streams split from the original source.

    The app produces overlapping sliding-window diarization segments.  Copying
    those windows directly would duplicate audio.  Instead this function builds
    a sample-accurate decision mask from segment start/end times:

      * speech windows vote for Speaker 1 or Speaker 2;
      * non-speech windows vote for silence;
      * ties between speakers are resolved by the newest active window;
      * each source sample is copied to at most one speaker stream.

    The two returned WAV streams have the same duration as the source audio,
    with silence where the other speaker/non-speech was selected.
    """
    audio = np.asarray(audio, dtype=np.float32).flatten()
    n = int(audio.size)
    streams = np.zeros((NUM_SPEAKERS, n), dtype=np.float32)

    stats = {
        "duration_sec": n / float(sample_rate) if sample_rate else 0.0,
        "speech_segments": 0,
        "non_speech_segments": 0,
        "ignored_segments": 0,
        "speaker_samples": [0 for _ in range(NUM_SPEAKERS)],
    }
    if n == 0:
        return streams, stats

    normalized = []
    for seg in segs:
        start = int(round(float(seg.start_time) * sample_rate))
        end = int(round(float(seg.end_time) * sample_rate))
        start = max(0, min(n, start))
        end = max(0, min(n, end))
        if end <= start:
            stats["ignored_segments"] += 1
            continue

        if seg.is_speech and isinstance(seg.spk_id, int) and 0 <= seg.spk_id < NUM_SPEAKERS:
            normalized.append((start, end, seg.spk_id))
            stats["speech_segments"] += 1
        else:
            normalized.append((start, end, None))
            stats["non_speech_segments"] += 1

    if not normalized:
        return streams, stats

    # Sort by time so "newest active window" tie-breaking is stable and based
    # on the diarization clock, not on asynchronous embedding completion order.
    normalized.sort(key=lambda x: (x[0], x[1], -1 if x[2] is None else x[2]))

    events = []
    for order, (start, end, spk_id) in enumerate(normalized):
        events.append((start, 1, spk_id, order))
        events.append((end, -1, spk_id, order))
    events.sort(key=lambda x: x[0])

    active_spk_orders = [[] for _ in range(NUM_SPEAKERS)]
    active_silence_orders = []

    def _remove_once(values, item):
        try:
            values.remove(item)
        except ValueError:
            pass

    def _choose_active_speaker():
        speech_counts = [len(active_spk_orders[i]) for i in range(NUM_SPEAKERS)]
        max_speech = max(speech_counts)
        if max_speech <= 0:
            return None

        # If non-speech windows outvote the best speaker window count, keep
        # this interval silent.  Equal vote keeps speech to avoid chopping words.
        if len(active_silence_orders) > max_speech:
            return None

        if speech_counts[0] > speech_counts[1]:
            return 0
        if speech_counts[1] > speech_counts[0]:
            return 1

        # Speaker tie: newest active diarization window wins.
        newest = None
        chosen = None
        for spk_id, orders in enumerate(active_spk_orders):
            if not orders:
                continue
            order = max(orders)
            if newest is None or order > newest:
                newest = order
                chosen = spk_id
        return chosen

    idx = 0
    while idx < len(events):
        sample_pos = events[idx][0]

        # Apply all starts/ends at this exact sample boundary before deciding
        # the interval [sample_pos, next_sample_pos).
        while idx < len(events) and events[idx][0] == sample_pos:
            _pos, delta, spk_id, order = events[idx]
            bucket = active_silence_orders if spk_id is None else active_spk_orders[spk_id]
            if delta > 0:
                bucket.append(order)
            else:
                _remove_once(bucket, order)
            idx += 1

        next_pos = events[idx][0] if idx < len(events) else n
        next_pos = max(sample_pos, min(n, next_pos))
        if next_pos <= sample_pos:
            continue

        spk_id = _choose_active_speaker()
        if spk_id is not None:
            streams[spk_id, sample_pos:next_pos] = audio[sample_pos:next_pos]
            stats["speaker_samples"][spk_id] += (next_pos - sample_pos)

    return streams, stats


class _AudioChunkCursor:
    """Forward-only sample reader over recorded diarization chunks."""

    def __init__(self, chunks):
        self.chunks = []
        for chunk in chunks:
            arr = np.asarray(chunk, dtype=np.float32).ravel()
            if arr.size:
                self.chunks.append(arr)
        self.chunk_idx = 0
        self.chunk_start = 0

    def seek_forward(self, sample_pos):
        sample_pos = int(max(0, sample_pos))
        if sample_pos < self.chunk_start:
            # The export path is expected to be monotonic.  Reset defensively if
            # a caller seeks backwards.
            self.chunk_idx = 0
            self.chunk_start = 0
        while self.chunk_idx < len(self.chunks):
            chunk_len = int(self.chunks[self.chunk_idx].size)
            if self.chunk_start + chunk_len > sample_pos:
                break
            self.chunk_start += chunk_len
            self.chunk_idx += 1

    def iter_range(self, start, end, max_block_samples=EXPORT_BLOCK_SAMPLES):
        start = int(start)
        end = int(end)
        pos = start
        self.seek_forward(pos)

        while pos < end:
            block_limit = min(end - pos, int(max_block_samples))
            if self.chunk_idx >= len(self.chunks):
                # Should not normally happen, but keeps export length exact even
                # if the recorded sample count and chunks disagree.
                yield np.zeros(block_limit, dtype=np.float32)
                pos += block_limit
                continue

            chunk = self.chunks[self.chunk_idx]
            offset = pos - self.chunk_start
            if offset < 0:
                offset = 0
            available = int(chunk.size) - int(offset)
            if available <= 0:
                self.chunk_start += int(chunk.size)
                self.chunk_idx += 1
                continue

            take = min(block_limit, available)
            yield chunk[offset:offset + take]
            pos += take

            if offset + take >= chunk.size:
                self.chunk_start += int(chunk.size)
                self.chunk_idx += 1


def _float32_to_pcm16_bytes(samples):
    samples = np.asarray(samples, dtype=np.float32).ravel()
    if samples.size == 0:
        return b""
    clipped = np.clip(samples, -1.0, 1.0)
    # Preserve the full negative int16 range while keeping positive peaks at
    # 32767.  astype('<i2') guarantees little-endian WAV PCM bytes.
    pcm = np.where(clipped < 0.0, clipped * 32768.0, clipped * 32767.0).astype('<i2')
    return pcm.tobytes()


def _zero_pcm16_bytes(num_samples):
    return b"\x00\x00" * int(max(0, num_samples))


def build_split_intervals(segs, total_samples, sample_rate=SAMPLE_RATE):
    """Build non-overlapping speaker/silence intervals from overlapping windows."""
    n = int(max(0, total_samples))
    stats = {
        "duration_sec": n / float(sample_rate) if sample_rate else 0.0,
        "speech_segments": 0,
        "non_speech_segments": 0,
        "ignored_segments": 0,
        "speaker_samples": [0 for _ in range(NUM_SPEAKERS)],
        "interval_count": 0,
    }
    if n <= 0:
        return [], stats

    normalized = []
    for seg in segs:
        start = int(round(float(seg.start_time) * sample_rate))
        end = int(round(float(seg.end_time) * sample_rate))
        start = max(0, min(n, start))
        end = max(0, min(n, end))
        if end <= start:
            stats["ignored_segments"] += 1
            continue

        if seg.is_speech and isinstance(seg.spk_id, int) and 0 <= seg.spk_id < NUM_SPEAKERS:
            normalized.append((start, end, int(seg.spk_id)))
            stats["speech_segments"] += 1
        else:
            normalized.append((start, end, None))
            stats["non_speech_segments"] += 1

    if not normalized:
        # No diarization decisions yet; both streams should be full-length silence.
        stats["interval_count"] = 1
        return [(0, n, None)], stats

    normalized.sort(key=lambda x: (x[0], x[1], -1 if x[2] is None else x[2]))

    events = []
    for order, (start, end, spk_id) in enumerate(normalized):
        events.append((start, 1, spk_id, order))
        events.append((end, -1, spk_id, order))
    events.sort(key=lambda x: x[0])

    active_spk_orders = [[] for _ in range(NUM_SPEAKERS)]
    active_silence_orders = []

    def _remove_once(values, item):
        try:
            values.remove(item)
        except ValueError:
            pass

    def _choose_active_speaker():
        speech_counts = [len(active_spk_orders[i]) for i in range(NUM_SPEAKERS)]
        max_speech = max(speech_counts)
        if max_speech <= 0:
            return None
        if len(active_silence_orders) > max_speech:
            return None
        if speech_counts[0] > speech_counts[1]:
            return 0
        if speech_counts[1] > speech_counts[0]:
            return 1

        newest = None
        chosen = None
        for spk_id, orders in enumerate(active_spk_orders):
            if not orders:
                continue
            order = max(orders)
            if newest is None or order > newest:
                newest = order
                chosen = spk_id
        return chosen

    intervals = []
    idx = 0
    curr_pos = 0
    while idx < len(events):
        sample_pos = events[idx][0]
        sample_pos = max(0, min(n, sample_pos))

        if sample_pos > curr_pos:
            spk_id = _choose_active_speaker()
            intervals.append((curr_pos, sample_pos, spk_id))
            if spk_id is not None:
                stats["speaker_samples"][spk_id] += sample_pos - curr_pos
            curr_pos = sample_pos

        while idx < len(events) and events[idx][0] == sample_pos:
            _pos, delta, spk_id, order = events[idx]
            bucket = active_silence_orders if spk_id is None else active_spk_orders[spk_id]
            if delta > 0:
                bucket.append(order)
            else:
                _remove_once(bucket, order)
            idx += 1

    if curr_pos < n:
        spk_id = _choose_active_speaker()
        intervals.append((curr_pos, n, spk_id))
        if spk_id is not None:
            stats["speaker_samples"][spk_id] += n - curr_pos

    # Merge adjacent intervals with the same decision to reduce write calls.
    merged = []
    for start, end, spk_id in intervals:
        if end <= start:
            continue
        if merged and merged[-1][2] == spk_id and merged[-1][1] == start:
            merged[-1] = (merged[-1][0], end, spk_id)
        else:
            merged.append((start, end, spk_id))
    stats["interval_count"] = len(merged)
    return merged, stats


# =====================================================================
# RealtimeSTT workers and live diarized stream splitter
# =====================================================================
class SpeakerSTTWorker(QThread):
    """Owns one RealtimeSTT AudioToTextRecorder for one diarized speaker."""

    ready = pyqtSignal(int)
    status = pyqtSignal(int, str)
    error = pyqtSignal(int, str)
    final_text = pyqtSignal(int, str)
    realtime_text = pyqtSignal(int, str)
    stabilized_text = pyqtSignal(int, str)

    def __init__(self, speaker_id, *, model, realtime_model, language, device, compute_type):
        super().__init__()
        self.speaker_id = int(speaker_id)
        self.model = model
        self.realtime_model = realtime_model
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.feed_queue = queue.Queue(maxsize=STT_FEED_QUEUE_MAX)
        self._running = True
        self._ready = False
        self._recorder = None
        self._text_thread = None
        self._stop_event = threading.Event()

    def is_ready(self):
        return bool(self._ready)

    def _enqueue_feed_bytes(self, pcm_bytes):
        try:
            self.feed_queue.put_nowait(bytes(pcm_bytes))
        except queue.Full:
            # Keep latency bounded. Drop the oldest chunk instead of blocking
            # the diarization/audio processing thread.
            try:
                self.feed_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.feed_queue.put_nowait(bytes(pcm_bytes))
            except queue.Full:
                pass

    def feed_audio_bytes(self, pcm_bytes):
        if not self._ready or not pcm_bytes:
            return
        raw = bytes(pcm_bytes)
        expected = int(STT_FEED_SAMPLES) * 2  # int16 PCM: 2 bytes/sample
        if expected <= 0:
            return

        # RealtimeSTT forwards one buffer to Silero VAD.  At 16 kHz that
        # buffer must be 512 samples in the currently used Silero model, so
        # never let a larger/smaller frame pass through unmodified.
        if len(raw) == expected:
            self._enqueue_feed_bytes(raw)
            return

        for offset in range(0, len(raw), expected):
            chunk = raw[offset:offset + expected]
            if not chunk:
                continue
            if len(chunk) < expected:
                chunk = chunk + (b"\x00" * (expected - len(chunk)))
            self._enqueue_feed_bytes(chunk)

    def run(self):
        try:
            self.status.emit(self.speaker_id, "loading RealtimeSTT…")
            from RealtimeSTT import AudioToTextRecorder

            def on_realtime_update(text):
                text = (text or "").strip()
                if text:
                    self.realtime_text.emit(self.speaker_id, text)

            def on_realtime_stabilized(text):
                text = (text or "").strip()
                if text:
                    self.stabilized_text.emit(self.speaker_id, text)

            self._recorder = AudioToTextRecorder(
                model=self.model,
                realtime_model_type=self.realtime_model,
                language=self.language,
                device=self.device,
                compute_type=self.compute_type,
                use_microphone=False,
                spinner=False,
                enable_realtime_transcription=True,
                use_main_model_for_realtime=True,
                realtime_processing_pause=0.2,
                init_realtime_after_seconds=0.2,
                on_realtime_transcription_update=on_realtime_update,
                on_realtime_transcription_stabilized=on_realtime_stabilized,
                post_speech_silence_duration=0.45,
                min_length_of_recording=0.35,
                min_gap_between_recordings=0.05,
                pre_recording_buffer_duration=0.35,
                silero_sensitivity=0.45,
                webrtc_sensitivity=3,
                beam_size=5,
                beam_size_realtime=3,
                buffer_size=STT_FEED_SAMPLES,
                sample_rate=SAMPLE_RATE,
                no_log_file=True,
                level=logging.ERROR,
            )
            self._ready = True
            self.ready.emit(self.speaker_id)
            self.status.emit(self.speaker_id, "ready")

            self._text_thread = threading.Thread(target=self._text_loop, daemon=True)
            self._text_thread.start()

            while self._running and not self._stop_event.is_set():
                try:
                    data = self.feed_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if data is None:
                    break
                try:
                    self._recorder.feed_audio(data)
                except Exception as exc:
                    self.error.emit(self.speaker_id, f"feed_audio error: {exc}")
                    time.sleep(0.05)
        except Exception as exc:
            self.error.emit(self.speaker_id, str(exc))
        finally:
            self._ready = False
            self._stop_event.set()
            try:
                if self._recorder is not None:
                    try:
                        self._recorder.abort()
                    except Exception:
                        pass
                    self._recorder.shutdown()
            except Exception as exc:
                self.error.emit(self.speaker_id, f"shutdown error: {exc}")
            self.status.emit(self.speaker_id, "stopped")

    def _text_loop(self):
        def process_text(text):
            text = (text or "").strip()
            if text:
                self.final_text.emit(self.speaker_id, text)

        while self._running and not self._stop_event.is_set():
            try:
                self._recorder.text(process_text)
            except Exception as exc:
                if self._running and not self._stop_event.is_set():
                    self.error.emit(self.speaker_id, f"text() error: {exc}")
                    time.sleep(0.2)

    def stop(self):
        self._running = False
        self._stop_event.set()
        try:
            self.feed_queue.put_nowait(None)
        except Exception:
            pass
        try:
            if self._recorder is not None:
                self._recorder.abort()
        except Exception:
            pass


class RealtimeDiarizedSTTSplitter:
    """Online version of the WAV splitter used for RealtimeSTT feed."""

    def __init__(self, stt_workers, *, sample_rate=SAMPLE_RATE,
                 feed_samples=STT_FEED_SAMPLES,
                 commit_delay_sec=STT_SPLIT_COMMIT_DELAY_SEC):
        self.stt_workers = list(stt_workers or [])
        self.sample_rate = int(sample_rate)
        self.feed_samples = int(feed_samples)
        self.commit_delay_samples = int(float(commit_delay_sec) * self.sample_rate)
        self.lock = threading.RLock()
        self.active = False
        self.reset()

    def set_active(self, active):
        with self.lock:
            self.active = bool(active)

    def reset(self):
        with getattr(self, "lock", threading.RLock()):
            self.chunks = []
            self.total_samples = 0
            self.committed_samples = 0
            self.segments = []
            self.next_order = 0
            self.final_silence_sent = False

    def add_audio(self, audio_chunk):
        arr = np.asarray(audio_chunk, dtype=np.float32).ravel().copy()
        if arr.size == 0:
            return
        with self.lock:
            start = self.total_samples
            self.chunks.append((start, arr))
            self.total_samples += int(arr.size)
            if self.active:
                self._flush_locked(final=False)

    def add_segment(self, seg):
        start = max(0, int(round(float(seg.start_time) * self.sample_rate)))
        end = max(start, int(round(float(seg.end_time) * self.sample_rate)))
        spk_id = None
        is_speech = bool(seg.is_speech)
        if is_speech and isinstance(seg.spk_id, int) and 0 <= seg.spk_id < NUM_SPEAKERS:
            spk_id = int(seg.spk_id)
        with self.lock:
            self.next_order += 1
            self.segments.append((start, end, spk_id, is_speech, self.next_order))
            self._prune_segments_locked()
            if self.active:
                self._flush_locked(final=False)

    def flush(self, final=False):
        with self.lock:
            final = bool(final)
            self._flush_locked(final=final)
            if final and not self.final_silence_sent:
                self._feed_silence_locked(int(STT_FINAL_SILENCE_SEC * self.sample_rate))
                self.final_silence_sent = True

    def _flush_locked(self, final=False):
        if not self.chunks:
            return
        target = self.total_samples if final else max(0, self.total_samples - self.commit_delay_samples)
        while self.committed_samples < target:
            remaining = target - self.committed_samples
            if remaining < self.feed_samples and not final:
                break
            source_len = min(self.feed_samples, max(0, self.total_samples - self.committed_samples))
            if source_len <= 0:
                break
            block = self._read_samples_locked(self.committed_samples, self.committed_samples + source_len)
            if source_len < self.feed_samples:
                block = np.pad(block, (0, self.feed_samples - source_len)).astype(np.float32, copy=False)
            owners = self._decide_owner_mask_locked(self.committed_samples, self.feed_samples)
            for speaker_id, worker in enumerate(self.stt_workers[:NUM_SPEAKERS]):
                out = np.zeros(self.feed_samples, dtype=np.float32)
                selected = owners == speaker_id
                if np.any(selected):
                    out[selected] = block[selected]
                worker.feed_audio_bytes(_float32_to_pcm16_bytes(out))
            self.committed_samples += source_len
            self._drop_consumed_audio_locked()
            self._prune_segments_locked()
            if source_len < self.feed_samples:
                break

    def _read_samples_locked(self, start, end):
        start = int(start)
        end = int(end)
        pieces = []
        pos = start
        for chunk_start, chunk in self.chunks:
            chunk_end = chunk_start + int(chunk.size)
            if chunk_end <= pos:
                continue
            if chunk_start >= end:
                break
            a = max(pos, chunk_start)
            b = min(end, chunk_end)
            if b > a:
                pieces.append(chunk[a - chunk_start:b - chunk_start])
                pos = b
            if pos >= end:
                break
        if pos < end:
            pieces.append(np.zeros(end - pos, dtype=np.float32))
        if not pieces:
            return np.zeros(end - start, dtype=np.float32)
        return np.concatenate(pieces).astype(np.float32, copy=False)

    def _decide_owner_mask_locked(self, block_start, n):
        block_start = int(block_start)
        block_end = block_start + int(n)
        votes = np.zeros((NUM_SPEAKERS, n), dtype=np.int16)
        silence_votes = np.zeros(n, dtype=np.int16)
        latest_owner = np.full(n, -1, dtype=np.int16)
        latest_order = np.full(n, -1, dtype=np.int32)

        for start, end, spk_id, is_speech, order in self.segments:
            if end <= block_start or start >= block_end:
                continue
            a = max(start, block_start) - block_start
            b = min(end, block_end) - block_start
            if b <= a:
                continue
            if is_speech and spk_id is not None:
                votes[spk_id, a:b] += 1
                newer = order >= latest_order[a:b]
                if np.any(newer):
                    owner_view = latest_owner[a:b]
                    order_view = latest_order[a:b]
                    owner_view[newer] = spk_id
                    order_view[newer] = order
            else:
                silence_votes[a:b] += 1
                newer = order >= latest_order[a:b]
                if np.any(newer):
                    owner_view = latest_owner[a:b]
                    order_view = latest_order[a:b]
                    owner_view[newer] = -1
                    order_view[newer] = order

        owners = np.full(n, -1, dtype=np.int16)
        v0 = votes[0]
        v1 = votes[1]
        max_spk = np.maximum(v0, v1)
        owners[(v0 > v1) & (v0 > silence_votes)] = 0
        owners[(v1 > v0) & (v1 > silence_votes)] = 1
        tied = (owners < 0) & (max_spk > 0) & (max_spk >= silence_votes) & (latest_owner >= 0)
        owners[tied] = latest_owner[tied]
        return owners

    def _feed_silence_locked(self, total_samples):
        total_samples = int(max(0, total_samples))
        if total_samples <= 0:
            return
        silence = _float32_to_pcm16_bytes(np.zeros(self.feed_samples, dtype=np.float32))
        chunks = int(np.ceil(total_samples / float(self.feed_samples)))
        for _ in range(chunks):
            for worker in self.stt_workers[:NUM_SPEAKERS]:
                worker.feed_audio_bytes(silence)

    def _drop_consumed_audio_locked(self):
        while self.chunks:
            chunk_start, chunk = self.chunks[0]
            if chunk_start + int(chunk.size) <= self.committed_samples:
                self.chunks.pop(0)
            else:
                break

    def _prune_segments_locked(self):
        keep_after = max(0, self.committed_samples - int(WIN_SAMPLES * 2))
        self.segments = [s for s in self.segments if s[1] >= keep_after]


class SplitWavExportWorker(QThread):
    status = pyqtSignal(str)
    export_done = pyqtSignal(dict)
    export_error = pyqtSignal(str)

    def __init__(self, audio_chunks, total_samples, segs, out_paths, sample_rate=SAMPLE_RATE):
        super().__init__()
        self.audio_chunks = tuple(audio_chunks)
        self.total_samples = int(total_samples)
        self.segs = list(segs)
        self.out_paths = list(out_paths)
        self.sample_rate = int(sample_rate)

    def run(self):
        try:
            if self.total_samples <= 0:
                raise RuntimeError("No diarization source audio was captured.")
            if len(self.out_paths) != NUM_SPEAKERS:
                raise RuntimeError("Invalid export output paths.")

            self.status.emit("Preparing diarization export map…")
            intervals, stats = build_split_intervals(
                self.segs, self.total_samples, self.sample_rate
            )
            stats["output_paths"] = self.out_paths

            self.status.emit("Writing Speaker 1/2 WAV files…")
            self._write_wavs(intervals)
            self.status.emit("Export complete.")
            self.export_done.emit(stats)
        except Exception as exc:
            self.export_error.emit(str(exc))

    def _write_wavs(self, intervals):
        for path in self.out_paths:
            folder = os.path.dirname(path)
            if folder:
                os.makedirs(folder, exist_ok=True)

        reader = _AudioChunkCursor(self.audio_chunks)
        written_samples = 0
        last_status_time = time.time()
        writers = []
        try:
            for path in self.out_paths:
                wf = wave.open(path, "wb")
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16 PCM
                wf.setframerate(self.sample_rate)
                wf.setnframes(self.total_samples)
                writers.append(wf)

            for start, end, spk_id in intervals:
                if self.isInterruptionRequested():
                    raise RuntimeError("WAV export was cancelled.")

                start = int(start)
                end = int(end)
                if end <= start:
                    continue

                if spk_id is None:
                    pos = start
                    while pos < end:
                        if self.isInterruptionRequested():
                            raise RuntimeError("WAV export was cancelled.")
                        block_len = min(int(EXPORT_BLOCK_SAMPLES), end - pos)
                        zeros = _zero_pcm16_bytes(block_len)
                        for wf in writers:
                            wf.writeframesraw(zeros)
                        pos += block_len
                        written_samples += block_len
                        last_status_time = self._emit_progress_if_needed(
                            written_samples, last_status_time
                        )
                    continue

                for samples in reader.iter_range(start, end, EXPORT_BLOCK_SAMPLES):
                    if self.isInterruptionRequested():
                        raise RuntimeError("WAV export was cancelled.")
                    block_len = int(samples.size)
                    if block_len <= 0:
                        continue
                    pcm = _float32_to_pcm16_bytes(samples)
                    zeros = _zero_pcm16_bytes(block_len)
                    if spk_id == 0:
                        writers[0].writeframesraw(pcm)
                        writers[1].writeframesraw(zeros)
                    else:
                        writers[0].writeframesraw(zeros)
                        writers[1].writeframesraw(pcm)
                    written_samples += block_len
                    last_status_time = self._emit_progress_if_needed(
                        written_samples, last_status_time
                    )
        finally:
            for wf in writers:
                try:
                    wf.close()
                except Exception:
                    pass

    def _emit_progress_if_needed(self, written_samples, last_status_time):
        now = time.time()
        if now - last_status_time >= EXPORT_STATUS_INTERVAL_SEC:
            pct = 100.0 * min(max(written_samples, 0), self.total_samples) / max(1, self.total_samples)
            self.status.emit(f"Exporting WAVs… {pct:.1f}%")
            return now
        return last_status_time


# =====================================================================
# Main window
# =====================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Binary Speaker Diarization (Enroll → Classify)")

        self.encoder = None
        self.vad_processor = None
        self.audio_capture = None
        self.audio_proc = None
        self.file_source = None
        self.export_worker = None
        self.stt_workers = []
        self.stt_splitter = None
        self.stt_ready = [False, False]
        self.stt_errors = ["", ""]

        self.spk_handler = SpeakerHandler()
        self.tl_manager = Timeline()

        self.is_diarizing = False
        self.finalizing_diarization = False
        self.final_diarization_window_requested = False
        self.pending_export_timed_out = False
        self.enrolling_speaker = None      # None / 0 / 1 (device enrolment)
        self.enrolling_from_file = False   # True iff currently enrolling via file
        self.last_export_paths = []
        self.export_dir = self._default_export_dir()

        self.models_loaded = {"encoder": False, "vad": False}

        self._setup_ui()
        QTimer.singleShot(500, self._init_app)

    # -----------------------------------------------------------------
    # UI setup
    # -----------------------------------------------------------------
    def _setup_ui(self):
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)

        self.status_label = QLabel("Preparing...")
        layout.addWidget(self.status_label)

        # Timeline
        self.scroll_area = QScrollArea()
        self.timeline = TimelineUI()
        self.scroll_area.setWidget(self.timeline)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(self.scroll_area, 1)

        # Step 1: Enrollment
        enroll_group = QGroupBox("Step 1.  Enroll Speakers (record from device until Stop, OR load a file)")
        enroll_layout = QVBoxLayout(enroll_group)

        self.s1_status = QLabel("Speaker 1: not enrolled")
        self.s1_status.setStyleSheet(f"color: {SPEAKER_COLORS[0]}; font-weight: bold; font-size: 14px;")
        s1_btns = QHBoxLayout()
        s1_btns.addWidget(self.s1_status, 1)
        self.s1_record_btn = QPushButton("Record Speaker 1 from Device")
        self.s1_record_btn.clicked.connect(lambda: self._toggle_enroll_record(0))
        s1_btns.addWidget(self.s1_record_btn)
        self.s1_file_btn = QPushButton("From File…")
        self.s1_file_btn.clicked.connect(lambda: self._enroll_from_file(0))
        s1_btns.addWidget(self.s1_file_btn)
        self.s1_clear_btn = QPushButton("Clear")
        self.s1_clear_btn.clicked.connect(lambda: self._clear_speaker(0))
        s1_btns.addWidget(self.s1_clear_btn)
        enroll_layout.addLayout(s1_btns)

        self.s2_status = QLabel("Speaker 2: not enrolled")
        self.s2_status.setStyleSheet(f"color: {SPEAKER_COLORS[1]}; font-weight: bold; font-size: 14px;")
        s2_btns = QHBoxLayout()
        s2_btns.addWidget(self.s2_status, 1)
        self.s2_record_btn = QPushButton("Record Speaker 2 from Device")
        self.s2_record_btn.clicked.connect(lambda: self._toggle_enroll_record(1))
        s2_btns.addWidget(self.s2_record_btn)
        self.s2_file_btn = QPushButton("From File…")
        self.s2_file_btn.clicked.connect(lambda: self._enroll_from_file(1))
        s2_btns.addWidget(self.s2_file_btn)
        self.s2_clear_btn = QPushButton("Clear")
        self.s2_clear_btn.clicked.connect(lambda: self._clear_speaker(1))
        s2_btns.addWidget(self.s2_clear_btn)
        enroll_layout.addLayout(s2_btns)

        layout.addWidget(enroll_group)

        # Step 2: Diarization
        dia_group = QGroupBox("Step 2.  Live Binary Diarization")
        dia_row = QHBoxLayout(dia_group)
        self.start_btn = QPushButton("Start Diarization")
        self.start_btn.clicked.connect(self._toggle_diarization)
        dia_row.addWidget(self.start_btn)
        self.reset_btn = QPushButton("Reset Timeline")
        self.reset_btn.clicked.connect(self._reset_tl)
        dia_row.addWidget(self.reset_btn)
        dia_row.addStretch()
        layout.addWidget(dia_group)

        # RealtimeSTT output
        stt_group = QGroupBox("RealtimeSTT per Speaker")
        stt_layout = QVBoxLayout(stt_group)
        self.stt_status_label = QLabel("STT: preparing…")
        stt_layout.addWidget(self.stt_status_label)

        self.s1_realtime_label = QLabel("Speaker 1 realtime: -")
        self.s1_realtime_label.setStyleSheet(f"color: {SPEAKER_COLORS[0]}; font-weight: bold;")
        stt_layout.addWidget(self.s1_realtime_label)
        self.s1_text_box = QPlainTextEdit()
        self.s1_text_box.setReadOnly(True)
        self.s1_text_box.setMaximumHeight(90)
        stt_layout.addWidget(self.s1_text_box)

        self.s2_realtime_label = QLabel("Speaker 2 realtime: -")
        self.s2_realtime_label.setStyleSheet(f"color: {SPEAKER_COLORS[1]}; font-weight: bold;")
        stt_layout.addWidget(self.s2_realtime_label)
        self.s2_text_box = QPlainTextEdit()
        self.s2_text_box.setReadOnly(True)
        self.s2_text_box.setMaximumHeight(90)
        stt_layout.addWidget(self.s2_text_box)

        layout.addWidget(stt_group)

        # Export folder selection is kept outside the End button path.
        # This avoids opening a native save dialog while diarization threads are
        # being finalized, which can make some Windows/Qt combinations appear
        # as "Not responding".
        export_group = QGroupBox("WAV Export Location")
        export_row = QHBoxLayout(export_group)
        export_row.addWidget(QLabel("Folder:"))
        self.export_dir_label = QLabel(self.export_dir)
        self.export_dir_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.export_dir_label.setWordWrap(True)
        export_row.addWidget(self.export_dir_label, 1)
        self.export_dir_btn = QPushButton("Change Folder…")
        self.export_dir_btn.clicked.connect(self._choose_export_dir)
        export_row.addWidget(self.export_dir_btn)
        layout.addWidget(export_group)

        # Audio device selection
        device_group = QGroupBox("Audio Device")
        dev_layout = QHBoxLayout(device_group)
        dev_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(300)
        dev_layout.addWidget(self.device_combo)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        dev_layout.addWidget(self.refresh_btn)
        self.apply_dev_btn = QPushButton("Apply")
        self.apply_dev_btn.clicked.connect(self._apply_device_change)
        dev_layout.addWidget(self.apply_dev_btn)
        dev_layout.addStretch()
        layout.addWidget(device_group)

        self._refresh_devices()

        # Auto-scroll timer
        self.scroll_timer = QTimer()
        self.scroll_timer.timeout.connect(self._auto_scroll)
        self.scroll_timer.start(2000)

        # Disable everything until models load
        self._set_all_buttons_enabled(False)

        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #2D2D30; color: #CCCCCC; }
            QPushButton { background: #3F3F46; color: #EEEEEE; border: 1px solid #555555;
                          padding: 8px 14px; margin: 4px; }
            QPushButton:hover { background: #505059; }
            QPushButton:disabled { background: #333337; color: #777777; }
            QLabel { padding: 4px; font-size: 13px; }
            QScrollArea { border: 1px solid #555555; }
            QGroupBox { font-weight: bold; border: 2px solid #555555; margin: 8px 0;
                        padding-top: 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QComboBox { background: #3F3F46; color: #EEEEEE; border: 1px solid #555555;
                        padding: 5px 10px; margin: 4px; }
            QPlainTextEdit { background: #1E1E1E; color: #EEEEEE; border: 1px solid #555555;
                             padding: 6px; font-size: 12px; }
        """)

    def _set_all_buttons_enabled(self, enabled):
        for btn in [
            self.s1_record_btn, self.s2_record_btn,
            self.s1_file_btn, self.s2_file_btn,
            self.s1_clear_btn, self.s2_clear_btn,
            self.start_btn, self.reset_btn, self.apply_dev_btn,
            self.export_dir_btn,
        ]:
            btn.setEnabled(enabled)

    # -----------------------------------------------------------------
    # Export folder
    # -----------------------------------------------------------------
    def _default_export_dir(self):
        home = os.path.expanduser("~")
        desktop = os.path.join(home, "Desktop")
        base = desktop if os.path.isdir(desktop) else home
        return os.path.join(base, "diarization_exports")

    def _choose_export_dir(self):
        if self._busy(check_diarize=True, check_enroll=True, check_file=True):
            QMessageBox.warning(self, "Busy", "Stop enrollment / diarization first.")
            return

        os.makedirs(self.export_dir, exist_ok=True)
        dialog = QFileDialog(self, "Select WAV export folder", self.export_dir)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)

        if dialog.exec():
            selected = dialog.selectedFiles()
            if selected and selected[0]:
                self._set_export_dir(selected[0])

    def _select_audio_file(self, title):
        dialog = QFileDialog(self, title, "")
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("Audio files (*.wav *.mp3 *.flac *.m4a *.ogg *.aac);;All files (*.*)")
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        if dialog.exec():
            selected = dialog.selectedFiles()
            if selected:
                return selected[0]
        return ""

    def _set_export_dir(self, folder):
        folder = os.path.abspath(os.path.expanduser(str(folder)))
        self.export_dir = folder
        self.export_dir_label.setText(folder)
        self.status_label.setText(f"WAV export folder: {folder}")

    def _make_export_base_path(self):
        os.makedirs(self.export_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.export_dir, f"diarization_split_{ts}")

    # -----------------------------------------------------------------
    # Device list
    # -----------------------------------------------------------------
    def _refresh_devices(self):
        self.device_combo.clear()
        try:
            for mic in sc.all_microphones():
                self.device_combo.addItem(f"[Mic] {mic.name}", (mic.name, True))
            for spk in sc.all_speakers():
                self.device_combo.addItem(f"[Speaker-loopback] {spk.name}", (spk.name, False))
        except Exception as e:
            print(f"Error listing audio devices: {e}")
        if self.device_combo.count() > 0:
            try:
                default_spk = str(sc.default_speaker().name)
                for i in range(self.device_combo.count()):
                    name, is_mic = self.device_combo.itemData(i)
                    if (not is_mic) and name == default_spk:
                        self.device_combo.setCurrentIndex(i)
                        break
                else:
                    self.device_combo.setCurrentIndex(0)
            except Exception:
                self.device_combo.setCurrentIndex(0)

    def _apply_device_change(self):
        if self._busy(check_diarize=True, check_enroll=True, check_file=True):
            QMessageBox.warning(self, "Busy", "Stop enrollment / diarization first.")
            return
        if self.device_combo.currentIndex() < 0:
            return
        device_name, use_mic = self.device_combo.currentData()
        if self.audio_capture:
            self.audio_capture.stop()
            self.audio_capture.wait()
        self.audio_capture = AudioCapture(device_name=device_name, use_mic=use_mic)
        self.audio_capture.chunk_ready.connect(self.audio_proc.add_chunk)
        self.audio_capture.start()
        kind = "Mic" if use_mic else "Speaker-loopback"
        self.status_label.setText(f"Audio device: [{kind}] {device_name}")

    def _auto_scroll(self):
        if self.is_diarizing:
            sb = self.scroll_area.horizontalScrollBar()
            sb.setValue(sb.maximum())

    # -----------------------------------------------------------------
    # Init / model loading
    # -----------------------------------------------------------------
    def _init_app(self):
        self.resize(1400, 800)
        device = "cuda" if DEVICE_PREF == "cuda" and torch.cuda.is_available() else "cpu"
        if DEVICE_PREF == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.")
        else:
            print(f"Using {device.upper()} device.")

        self.encoder = WeSpeakerEncoder(device)
        self.encoder.model_loaded.connect(self._on_encoder_loaded)
        self.encoder.start()

        self.vad_processor = SileroVAD(VAD_THRESH)
        self.vad_processor.model_loaded.connect(self._on_vad_loaded)
        self.vad_processor.start()

        self.status_label.setText("Loading models (WeSpeaker + Silero VAD)…")

    def _on_encoder_loaded(self):
        self.models_loaded["encoder"] = True
        self._check_models_ready()

    def _on_vad_loaded(self):
        self.models_loaded["vad"] = True
        self._check_models_ready()

    def _check_models_ready(self):
        if not all(self.models_loaded.values()):
            return

        device_name, use_mic = None, False
        if self.device_combo.currentIndex() >= 0:
            device_name, use_mic = self.device_combo.currentData()
        self.audio_capture = AudioCapture(device_name=device_name, use_mic=use_mic)

        self.audio_proc = AudioProcessor(
            self.encoder, self.vad_processor, self.spk_handler, self.tl_manager
        )
        self.audio_capture.chunk_ready.connect(self.audio_proc.add_chunk)
        self.audio_proc.tl_updated.connect(self.timeline.update_segs)
        self.audio_proc.enrollment_progress.connect(self._on_enrollment_progress)

        self.audio_proc.start()
        self.audio_capture.start()
        self._init_realtime_stt()

        # Enable enrollment buttons; Start stays disabled until both speakers
        # are enrolled.
        for btn in [
            self.s1_record_btn, self.s2_record_btn,
            self.s1_file_btn, self.s2_file_btn,
            self.s1_clear_btn, self.s2_clear_btn,
            self.reset_btn, self.apply_dev_btn, self.export_dir_btn,
        ]:
            btn.setEnabled(True)
        self._update_start_button_state()

        self.status_label.setText("Ready. Enroll Speaker 1 and Speaker 2 to begin.")

    # -----------------------------------------------------------------
    # RealtimeSTT setup / UI updates
    # -----------------------------------------------------------------
    def _init_realtime_stt(self):
        if not REALTIME_STT_ENABLED:
            self.stt_status_label.setText("STT: disabled")
            return

        device = "cuda" if STT_DEVICE_PREF == "cuda" and torch.cuda.is_available() else "cpu"
        compute_type = STT_COMPUTE_TYPE or "default"
        self.stt_status_label.setText(
            f"STT: loading two RealtimeSTT recorders ({STT_MODEL}, {device})…"
        )

        self.stt_workers = []
        self.stt_ready = [False, False]
        self.stt_errors = ["", ""]
        for speaker_id in range(NUM_SPEAKERS):
            worker = SpeakerSTTWorker(
                speaker_id,
                model=STT_MODEL,
                realtime_model=STT_REALTIME_MODEL,
                language=STT_LANGUAGE,
                device=device,
                compute_type=compute_type,
            )
            worker.ready.connect(self._on_stt_ready)
            worker.status.connect(self._on_stt_status)
            worker.error.connect(self._on_stt_error)
            worker.final_text.connect(self._on_stt_final_text)
            worker.realtime_text.connect(self._on_stt_realtime_text)
            worker.stabilized_text.connect(self._on_stt_stabilized_text)
            worker.start()
            self.stt_workers.append(worker)

        self.stt_splitter = RealtimeDiarizedSTTSplitter(
            self.stt_workers,
            sample_rate=SAMPLE_RATE,
            feed_samples=STT_FEED_SAMPLES,
            commit_delay_sec=STT_SPLIT_COMMIT_DELAY_SEC,
        )
        self.audio_proc.set_stt_splitter(self.stt_splitter)

    def _stt_ready_all(self):
        return (not REALTIME_STT_ENABLED) or all(self.stt_ready)

    def _on_stt_ready(self, speaker_id):
        if 0 <= speaker_id < NUM_SPEAKERS:
            self.stt_ready[speaker_id] = True
        ready_count = sum(1 for ok in self.stt_ready if ok)
        self.stt_status_label.setText(f"STT: {ready_count}/{NUM_SPEAKERS} recorders ready")
        if ready_count == NUM_SPEAKERS:
            self.stt_status_label.setText(
                f"STT: ready. Streaming split delay ≈ {STT_SPLIT_COMMIT_DELAY_SEC:.1f}s"
            )

    def _on_stt_status(self, speaker_id, text):
        if not text:
            return
        if not self._stt_ready_all():
            self.stt_status_label.setText(f"STT Speaker {speaker_id + 1}: {text}")

    def _on_stt_error(self, speaker_id, message):
        if 0 <= speaker_id < NUM_SPEAKERS:
            self.stt_errors[speaker_id] = message
            self.stt_ready[speaker_id] = False
        self.stt_status_label.setText(
            f"STT Speaker {speaker_id + 1} error: {message}"
        )
        print(f"RealtimeSTT Speaker {speaker_id + 1} error: {message}")

    def _reset_stt_output(self):
        self.s1_realtime_label.setText("Speaker 1 realtime: -")
        self.s2_realtime_label.setText("Speaker 2 realtime: -")
        self.s1_text_box.clear()
        self.s2_text_box.clear()

    def _on_stt_realtime_text(self, speaker_id, text):
        label = self.s1_realtime_label if speaker_id == 0 else self.s2_realtime_label
        label.setText(f"Speaker {speaker_id + 1} realtime: {text}")

    def _on_stt_stabilized_text(self, speaker_id, text):
        label = self.s1_realtime_label if speaker_id == 0 else self.s2_realtime_label
        label.setText(f"Speaker {speaker_id + 1} stable: {text}")

    def _on_stt_final_text(self, speaker_id, text):
        box = self.s1_text_box if speaker_id == 0 else self.s2_text_box
        stamp = datetime.now().strftime("%H:%M:%S")
        box.appendPlainText(f"[{stamp}] {text}")
        sb = box.verticalScrollBar()
        sb.setValue(sb.maximum())
        print(f"STT Speaker {speaker_id + 1}: {text}")

    # -----------------------------------------------------------------
    # Busy-state helper
    # -----------------------------------------------------------------
    def _busy(self, check_diarize=True, check_enroll=True, check_file=True):
        if self.finalizing_diarization:
            return True
        if check_diarize and self.is_diarizing:
            return True
        if check_enroll and self.enrolling_speaker is not None and not self.enrolling_from_file:
            return True
        if check_file and self.enrolling_from_file:
            return True
        return False

    # -----------------------------------------------------------------
    # Enrollment from device (toggle)
    # -----------------------------------------------------------------
    def _toggle_enroll_record(self, speaker_id):
        # If already recording for this speaker → stop & finalize.
        if self.enrolling_speaker == speaker_id and not self.enrolling_from_file:
            self._stop_device_enrollment(finalize=True)
            return

        if self._busy():
            QMessageBox.warning(self, "Busy", "Stop the current operation first.")
            return

        # Reset previous embeddings for this speaker (re-enroll from scratch).
        self.spk_handler.reset_speaker(speaker_id)

        # Switch to the right enrollment phase BEFORE resuming capture so
        # set_phase() drains stale buffers / queues first.
        phase = PHASE_ENROLL_S1 if speaker_id == 0 else PHASE_ENROLL_S2
        self.audio_proc.set_phase(phase)
        self.audio_proc.resume()
        self.audio_capture.resume()

        self.enrolling_speaker = speaker_id
        self.enrolling_from_file = False

        self._set_speaker_status(
            speaker_id,
            f"Speaker {speaker_id + 1}: recording…  (0 segments)"
        )
        self.status_label.setText(
            f"Recording Speaker {speaker_id + 1} from device. "
            f"Click 'Stop (Speaker {speaker_id + 1})' when done."
        )
        self._update_buttons_during_enroll(speaker_id, recording=True, from_file=False)

    def _stop_device_enrollment(self, finalize=True):
        speaker_id = self.enrolling_speaker
        if speaker_id is None:
            return

        self.audio_capture.pause()
        self.audio_proc.pause()
        self.audio_proc.set_phase(PHASE_IDLE)

        if finalize:
            ok = self.spk_handler.finalize_enrollment(speaker_id)
            n = self.spk_handler.enrollment_count(speaker_id)
            if ok:
                self._set_speaker_status(
                    speaker_id,
                    f"Speaker {speaker_id + 1}: ✓ enrolled  ({n} segments)"
                )
                self.status_label.setText(
                    f"Speaker {speaker_id + 1} enrolled with {n} segments."
                )
            else:
                self._set_speaker_status(
                    speaker_id,
                    f"Speaker {speaker_id + 1}: enrollment failed (no speech)"
                )
                self.status_label.setText(
                    f"Speaker {speaker_id + 1}: no speech detected — try again."
                )

        self.enrolling_speaker = None
        self.enrolling_from_file = False
        self._update_buttons_during_enroll(speaker_id, recording=False)
        self._update_start_button_state()

    # -----------------------------------------------------------------
    # Enrollment from file
    # -----------------------------------------------------------------
    def _enroll_from_file(self, speaker_id):
        if self._busy():
            QMessageBox.warning(self, "Busy", "Stop the current operation first.")
            return

        file_path = self._select_audio_file(
            f"Select audio file for Speaker {speaker_id + 1}"
        )
        if not file_path:
            return

        self.spk_handler.reset_speaker(speaker_id)

        # Make sure the live capture is paused so it doesn't pollute
        # the buffer with mic audio while we play the file in.
        self.audio_capture.pause()

        phase = PHASE_ENROLL_S1 if speaker_id == 0 else PHASE_ENROLL_S2
        self.audio_proc.set_phase(phase)
        self.audio_proc.resume()

        self.file_source = AudioFileSource(file_path)
        self.file_source.chunk_ready.connect(self.audio_proc.add_chunk)
        self.file_source.finished_playing.connect(
            lambda sid=speaker_id: self._on_file_enrollment_done(sid)
        )

        self.enrolling_speaker = speaker_id
        self.enrolling_from_file = True

        self._set_speaker_status(
            speaker_id,
            f"Speaker {speaker_id + 1}: processing file…  (0 segments)"
        )
        self.status_label.setText(
            f"Processing file for Speaker {speaker_id + 1}…"
        )
        self._update_buttons_during_enroll(speaker_id, recording=True, from_file=True)

        self.file_source.start()

    def _on_file_enrollment_done(self, speaker_id):
        # Poll until the VAD/embedding pipeline has drained, then finalize.
        self._poll_drain(speaker_id, waited_ms=0)

    def _poll_drain(self, speaker_id, waited_ms):
        if self.audio_proc.pipeline_idle() or waited_ms >= FILE_DRAIN_MAX_MS:
            self._finalize_file_enrollment(speaker_id)
        else:
            QTimer.singleShot(
                FILE_DRAIN_POLL_MS,
                lambda: self._poll_drain(speaker_id, waited_ms + FILE_DRAIN_POLL_MS)
            )

    def _finalize_file_enrollment(self, speaker_id):
        if self.file_source is not None:
            self.file_source.stop()
            self.file_source.wait(2000)
            self.file_source = None

        self.audio_proc.pause()
        self.audio_proc.set_phase(PHASE_IDLE)

        ok = self.spk_handler.finalize_enrollment(speaker_id)
        n = self.spk_handler.enrollment_count(speaker_id)
        if ok:
            self._set_speaker_status(
                speaker_id,
                f"Speaker {speaker_id + 1}: ✓ enrolled from file  ({n} segments)"
            )
            self.status_label.setText(
                f"Speaker {speaker_id + 1} enrolled from file with {n} segments."
            )
        else:
            self._set_speaker_status(
                speaker_id,
                f"Speaker {speaker_id + 1}: file enrollment failed (no speech)"
            )
            self.status_label.setText(
                f"Speaker {speaker_id + 1}: no speech detected in file."
            )

        self.enrolling_speaker = None
        self.enrolling_from_file = False
        self._update_buttons_during_enroll(speaker_id, recording=False)
        self._update_start_button_state()

    # -----------------------------------------------------------------
    # Clear a speaker's enrollment
    # -----------------------------------------------------------------
    def _clear_speaker(self, speaker_id):
        if self.enrolling_speaker == speaker_id:
            QMessageBox.warning(self, "Busy", "Stop enrollment first.")
            return
        if self.is_diarizing:
            QMessageBox.warning(self, "Busy", "Stop diarization first.")
            return
        self.spk_handler.reset_speaker(speaker_id)
        self._set_speaker_status(speaker_id, f"Speaker {speaker_id + 1}: not enrolled")
        self.status_label.setText(f"Speaker {speaker_id + 1} cleared.")
        self._update_start_button_state()

    # -----------------------------------------------------------------
    # Diarization (start / end + export)
    # -----------------------------------------------------------------
    def _toggle_diarization(self):
        if self.finalizing_diarization:
            return
        if self.enrolling_speaker is not None or self.enrolling_from_file:
            QMessageBox.warning(self, "Busy", "Stop enrollment first.")
            return
        if not self.spk_handler.both_enrolled():
            QMessageBox.warning(self, "Not ready", "Both speakers must be enrolled first.")
            return
        if REALTIME_STT_ENABLED and not self._stt_ready_all():
            QMessageBox.warning(
                self,
                "STT not ready",
                "RealtimeSTT is still loading or failed. Check the STT status line/console."
            )
            return

        if self.is_diarizing:
            self._finish_diarization_and_export()
            return

        # Fresh start — reset timeline/source-audio/STT storage and switch phase.
        self.tl_manager.reset()
        self.timeline.update_segs([])
        self._reset_stt_output()
        self.audio_proc.set_phase(PHASE_DIARIZE)

        self.tl_manager.start_timeline()
        self.audio_proc.resume()
        self.audio_capture.resume()

        self.is_diarizing = True
        self.start_btn.setText("End Diarization + Export WAV")
        self.status_label.setText("Diarizing… split streams are feeding RealtimeSTT; click End Diarization + Export WAV to save WAVs.")
        self._set_enroll_buttons_enabled(False)

    def _finish_diarization_and_export(self):
        """Stop new capture, drain queued classification work, then export."""
        if self.finalizing_diarization:
            return
        self.finalizing_diarization = True

        # Stop receiving new source samples, but keep AudioProcessor running so
        # already-queued chunks/VAD/embedding results can still update timeline.
        self.audio_capture.pause()
        self.final_diarization_window_requested = False

        self.start_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self._set_enroll_buttons_enabled(False)
        self.status_label.setText("Finalizing diarization before WAV export…")

        QTimer.singleShot(
            DIARIZATION_DRAIN_POLL_MS,
            lambda: self._poll_diarization_drain(waited_ms=DIARIZATION_DRAIN_POLL_MS),
        )

    def _poll_diarization_drain(self, waited_ms):
        if not self.finalizing_diarization:
            return
        idle = self.audio_proc.pipeline_idle()
        min_waited = waited_ms >= DIARIZATION_DRAIN_MIN_MS
        timed_out = waited_ms >= DIARIZATION_DRAIN_MAX_MS

        # Once all already-queued chunks/windows have drained, ask the processor
        # for one final window.  This captures the tail after the last periodic
        # WINDOW_PROCESS_INTERVAL tick without racing against queued chunks.
        if min_waited and idle and not self.final_diarization_window_requested:
            self.final_diarization_window_requested = True
            self.audio_proc.request_final_window()
            QTimer.singleShot(
                DIARIZATION_DRAIN_POLL_MS,
                lambda: self._poll_diarization_drain(waited_ms + DIARIZATION_DRAIN_POLL_MS),
            )
            return

        if min_waited and ((idle and self.final_diarization_window_requested) or timed_out):
            self._finalize_diarization_and_export(timed_out=timed_out and not idle)
            return

        QTimer.singleShot(
            DIARIZATION_DRAIN_POLL_MS,
            lambda: self._poll_diarization_drain(waited_ms + DIARIZATION_DRAIN_POLL_MS),
        )

    def _finalize_diarization_and_export(self, timed_out=False):
        self.audio_proc.pause()
        self.tl_manager.pause_timeline()
        # Push the delayed tail of the live split streams into RealtimeSTT
        # before switching the processor back to IDLE/export cleanup.
        self.audio_proc.flush_realtime_stt()
        if self.stt_splitter is not None:
            self.stt_splitter.set_active(False)

        # Avoid forcing a heavy timeline repaint at the same moment export
        # starts.  The timeline has already been updated periodically during
        # diarization; export uses the timeline manager's segment list directly.
        self._begin_export_split_wavs(timed_out=timed_out)

    def _begin_export_split_wavs(self, timed_out=False):
        chunks, total_samples = self.audio_proc.get_diarization_audio_chunks_snapshot()
        segs = list(self.tl_manager.get_segs())

        if total_samples <= 0 or not chunks:
            self._complete_diarization_after_export(
                "Diarization stopped. No source audio was captured, so WAV export was skipped.",
                timed_out=timed_out,
            )
            return

        try:
            base = self._make_export_base_path()
        except Exception as exc:
            self._complete_diarization_after_export(
                f"Diarization stopped, but export folder could not be prepared: {exc}",
                timed_out=timed_out,
            )
            return

        out_paths = [f"{base}_speaker{i + 1}.wav" for i in range(NUM_SPEAKERS)]

        self.pending_export_timed_out = bool(timed_out)
        self.last_export_paths = []
        self.status_label.setText(f"Exporting WAVs to {self.export_dir} …")
        self._update_start_button_state()

        self.export_worker = SplitWavExportWorker(
            chunks, total_samples, segs, out_paths, SAMPLE_RATE
        )
        self.export_worker.status.connect(self._on_export_status)
        self.export_worker.export_done.connect(self._on_export_done)
        self.export_worker.export_error.connect(self._on_export_error)
        self.export_worker.finished.connect(self.export_worker.deleteLater)
        self.export_worker.start()

    def _on_export_status(self, text):
        if self.finalizing_diarization:
            self.status_label.setText(text)

    def _on_export_done(self, stats):
        out_paths = list(stats.get("output_paths", []))
        self.last_export_paths = out_paths
        saved_msg, details = self._format_export_summary(stats)
        print(details)
        self.export_worker = None
        self._complete_diarization_after_export(
            saved_msg,
            timed_out=self.pending_export_timed_out,
        )

    def _on_export_error(self, message):
        export_message = f"Diarization stopped, but WAV export failed: {message}"
        QMessageBox.critical(self, "Export failed", export_message)
        self.export_worker = None
        self._complete_diarization_after_export(
            export_message,
            timed_out=self.pending_export_timed_out,
        )

    def _format_export_summary(self, stats):
        out_paths = list(stats.get("output_paths", []))
        if len(out_paths) < NUM_SPEAKERS:
            out_paths = ["", ""]
        spk_samples = stats.get("speaker_samples", [0, 0])
        spk_secs = [float(spk_samples[i]) / SAMPLE_RATE for i in range(NUM_SPEAKERS)]
        duration = float(stats.get("duration_sec", 0.0))
        saved_msg = (
            f"Saved speaker WAVs: {os.path.basename(out_paths[0])}, "
            f"{os.path.basename(out_paths[1])}  "
            f"(source {duration:.2f}s, S1 {spk_secs[0]:.2f}s, S2 {spk_secs[1]:.2f}s)"
        )

        details = (
            f"Saved:\n{out_paths[0]}\n{out_paths[1]}\n\n"
            f"Source duration: {duration:.2f}s\n"
            f"Speaker 1 active: {spk_secs[0]:.2f}s\n"
            f"Speaker 2 active: {spk_secs[1]:.2f}s\n"
            f"Speech windows used: {stats.get('speech_segments', 0)}\n"
            f"Non-speech windows used: {stats.get('non_speech_segments', 0)}\n"
            f"Export intervals written: {stats.get('interval_count', 0)}"
        )
        if stats.get("speech_segments", 0) == 0:
            details += "\n\nNo speech windows were classified, so both files are time-aligned silence."
        return saved_msg, details

    def _complete_diarization_after_export(self, export_message, timed_out=False):
        # Cleanup after export/cancel/error.  set_phase(PHASE_IDLE) clears the
        # processor's source-audio list; the worker uses its own chunk snapshot.
        self.audio_proc.set_phase(PHASE_IDLE)
        self.is_diarizing = False
        self.finalizing_diarization = False
        self.final_diarization_window_requested = False
        self.pending_export_timed_out = False

        self._set_enroll_buttons_enabled(True)
        self.reset_btn.setEnabled(True)
        self._update_start_button_state()

        if timed_out:
            export_message = (export_message or "Diarization stopped.") + "  Warning: pipeline drain timed out."
        self.status_label.setText(export_message or "Diarization stopped.")

    def _reset_tl(self):
        if self.finalizing_diarization:
            QMessageBox.warning(self, "Busy", "WAV export finalization is still running.")
            return
        # Pause first if currently diarizing.
        if self.is_diarizing:
            self.audio_capture.pause()
            self.audio_proc.pause()
            self.tl_manager.pause_timeline()
            self.is_diarizing = False

        self.tl_manager.reset()
        self.timeline.update_segs([])
        self.audio_proc.set_phase(PHASE_IDLE)
        self._reset_stt_output()
        self.start_btn.setText("Start Diarization")
        self._set_enroll_buttons_enabled(True)
        self._update_start_button_state()
        self.status_label.setText("Timeline reset.")

    # -----------------------------------------------------------------
    # Enrollment progress signal
    # -----------------------------------------------------------------
    def _on_enrollment_progress(self, speaker_id, count):
        if self.enrolling_speaker != speaker_id:
            return
        verb = "processing file" if self.enrolling_from_file else "recording"
        self._set_speaker_status(
            speaker_id,
            f"Speaker {speaker_id + 1}: {verb}…  ({count} segments)"
        )

    # -----------------------------------------------------------------
    # Button / status helpers
    # -----------------------------------------------------------------
    def _set_speaker_status(self, speaker_id, text):
        (self.s1_status if speaker_id == 0 else self.s2_status).setText(text)

    def _update_buttons_during_enroll(self, speaker_id, recording, from_file=False):
        # The active speaker's record button toggles to "Stop" while recording
        # from a device. During file enrollment, no Stop button is shown
        # (the file plays through to its end).
        rec_btn = self.s1_record_btn if speaker_id == 0 else self.s2_record_btn
        if recording and not from_file:
            rec_btn.setText(f"Stop (Speaker {speaker_id + 1})")
            rec_btn.setEnabled(True)
        else:
            rec_btn.setText(f"Record Speaker {speaker_id + 1} from Device")
            rec_btn.setEnabled(True)

        if recording:
            # Disable everything else.
            other = 1 - speaker_id
            other_rec = self.s1_record_btn if other == 0 else self.s2_record_btn
            other_file = self.s1_file_btn if other == 0 else self.s2_file_btn
            other_clear = self.s1_clear_btn if other == 0 else self.s2_clear_btn
            this_file = self.s1_file_btn if speaker_id == 0 else self.s2_file_btn
            this_clear = self.s1_clear_btn if speaker_id == 0 else self.s2_clear_btn
            other_rec.setEnabled(False)
            other_file.setEnabled(False)
            other_clear.setEnabled(False)
            this_file.setEnabled(False)
            this_clear.setEnabled(False)
            self.start_btn.setEnabled(False)
            self.reset_btn.setEnabled(False)
            self.apply_dev_btn.setEnabled(False)
            self.export_dir_btn.setEnabled(False)
            # If this is a file enrollment, the record button should also be
            # disabled (no Stop semantics for file).
            if from_file:
                rec_btn.setEnabled(False)
        else:
            # Re-enable everything; Start stays gated on both_enrolled().
            self.s1_record_btn.setEnabled(True)
            self.s2_record_btn.setEnabled(True)
            self.s1_file_btn.setEnabled(True)
            self.s2_file_btn.setEnabled(True)
            self.s1_clear_btn.setEnabled(True)
            self.s2_clear_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)
            self.apply_dev_btn.setEnabled(True)
            self.export_dir_btn.setEnabled(True)
            self._update_start_button_state()

    def _set_enroll_buttons_enabled(self, enabled):
        for btn in [
            self.s1_record_btn, self.s2_record_btn,
            self.s1_file_btn, self.s2_file_btn,
            self.s1_clear_btn, self.s2_clear_btn,
            self.apply_dev_btn, self.export_dir_btn,
        ]:
            btn.setEnabled(enabled)

    def _update_start_button_state(self):
        if self.finalizing_diarization:
            self.start_btn.setEnabled(False)
            self.start_btn.setText("Finalizing + Exporting…")
            return
        ok = (
            self.spk_handler.both_enrolled()
            and not self.is_diarizing
            and self.enrolling_speaker is None
        )
        self.start_btn.setEnabled(ok or self.is_diarizing)
        # Keep proper label.
        if self.is_diarizing:
            self.start_btn.setText("End Diarization + Export WAV")
        else:
            self.start_btn.setText("Start Diarization")

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------
    def closeEvent(self, event):
        if self.export_worker is not None and self.export_worker.isRunning():
            QMessageBox.warning(self, "Busy", "WAV export is still running. Close is disabled until export finishes.")
            event.ignore()
            return
        if self.file_source:
            self.file_source.stop()
            self.file_source.wait(1000)
        if self.audio_capture:
            self.audio_capture.stop()
            self.audio_capture.wait(1000)
        if self.audio_proc:
            self.audio_proc.stop()
            self.audio_proc.wait(1000)
        for worker in self.stt_workers:
            try:
                worker.stop()
                worker.wait(2000)
            except Exception:
                pass
        if self.encoder:
            self.encoder.stop_proc()
            self.encoder.wait(1000)
        if self.vad_processor:
            self.vad_processor.stop_processing()
            self.vad_processor.wait(1000)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
