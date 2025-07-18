import soundcard as sc, numpy as np
import threading, time, sys, os, urllib.request, queue
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, 
                           QHBoxLayout, QScrollArea, QInputDialog, QMessageBox, QLabel,
                           QComboBox, QGroupBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen, QFont
import torch, torchaudio
from scipy.spatial.distance import cosine
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

SPEAKER_COLORS = [
    "#FF4444", "#44FF44", "#4444FF", "#FFFF44", "#FF44FF", 
    "#44FFFF", "#FF8844", "#FF009D", "#8844FF", "#FFAA44"
]
PENDING_COLOR = "#888888"
MAX_SPEAKERS = 10
TIMELINE_HEIGHT = 600
TIMELINE_UPDATE_INTERVAL = 0.3
SIZE_UPDATE_INTERVAL = 1.0

SAMPLE_RATE = 16000
CHUNK_SIZE = 2048
WINDOW_SIZE = 1.0
WINDOW_PROCESS_INTERVAL = 0.1
WIN_SAMPLES = int(SAMPLE_RATE * WINDOW_SIZE)


DEVICE_PREF = "cuda"
VAD_THRESH = 0.5 
PENDING_THRESHOLD = 0.3
EMBEDDING_UPDATE_THRESHOLD = 0.4

MIN_PENDING_SIZE = 30
AUTO_CLUSTER_DISTANCE_THRESHOLD = 0.6
MIN_CLUSTER_SIZE = 15

class SileroVAD(QThread):
    model_loaded = pyqtSignal()
    
    def __init__(self, device="cpu", threshold=0.5):
        super().__init__()
        self.device = "cpu"
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
            model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', 
                                        force_reload=False, onnx=False)
            self.vad_model = model.to("cpu")
            self.get_speech_ts = utils[0]
            self.model_loaded_flag = True
            print("Silero VAD model loaded successfully")
        except Exception as e:
            print(f"Error loading Silero VAD model: {e}")
            raise e
        
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
                speech_timestamps = self.get_speech_ts(
                    audio_tensor, self.vad_model, threshold=self.threshold, 
                    sampling_rate=sr, return_seconds=False
                )
            return len(speech_timestamps) > 0
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

class SpeechBrainEncoder(QThread):
    model_loaded = pyqtSignal()
    
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.model = None
        self.model_loaded_flag = False
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "speechbrain")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.emb_queue = queue.Queue()
        self.res_queue = queue.Queue()
        self._stop_proc = False
    
    def run(self):
        model_url = "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.ckpt"
        model_path = os.path.join(self.cache_dir, "embedding_model_ecapa.ckpt")
        if not os.path.exists(model_path):
            print(f"Downloading ECAPA-TDNN model to {model_path}...")
            urllib.request.urlretrieve(model_url, model_path)
        
        from speechbrain.pretrained import EncoderClassifier
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir=self.cache_dir, 
            run_opts={"device": self.device}
        )
        self.model_loaded_flag = True
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
        if not self.model_loaded_flag:
            return None
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad(): 
            emb = self.model.encode_batch(waveform)
        return emb.squeeze().cpu().numpy()
    
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

class SpeakerHandler:
    def __init__(self, max_spks=MAX_SPEAKERS, change_thresh=PENDING_THRESHOLD, min_pending=MIN_PENDING_SIZE):
        self.max_spks = max_spks
        self.change_thresh = change_thresh
        self.min_pending = min_pending
        self.curr_spk = None
        self.mean_embs = [None] * max_spks
        self.spk_embs = [[] for _ in range(max_spks)]
        self.active_spks = set()
        self.pending_embs = []
        self.pending_times = []
        self.pending_enabled = True
        self.embedding_update_enabled = True
        self.embedding_updated = None
        self.timeline_manager = None
    
    def set_embedding_callback(self, callback):
        self.embedding_updated = callback
    
    def set_timeline_manager(self, timeline_manager):
        self.timeline_manager = timeline_manager
    
    def classify_spk(self, emb, seg_time):
        if not self.active_spks and self.pending_enabled:
            self.pending_embs.append(emb)
            self.pending_times.append(seg_time)
            self._check_pending_promotion()
            return "pending", 0.0
        
        if not self.active_spks:
            self.spk_embs[0].append(emb)
            self.mean_embs[0] = emb
            self.active_spks.add(0)
            self.curr_spk = 0
            return 0, 1.0
        
        active_mean_embs = []
        active_spk_ids = []
        for spk_id in self.active_spks:
            if self.mean_embs[spk_id] is not None:
                active_mean_embs.append(self.mean_embs[spk_id])
                active_spk_ids.append(spk_id)
        
        if not active_mean_embs:
            self.spk_embs[0].append(emb)
            self.mean_embs[0] = emb
            self.active_spks.add(0)
            self.curr_spk = 0
            return 0, 1.0
        
        emb_norm = emb / np.linalg.norm(emb)
        mean_embs_matrix = np.array(active_mean_embs)
        mean_embs_norm = mean_embs_matrix / np.linalg.norm(mean_embs_matrix, axis=1, keepdims=True)
        similarities = np.dot(mean_embs_norm, emb_norm)
        
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        best_spk = active_spk_ids[best_idx]
        
        if best_sim >= EMBEDDING_UPDATE_THRESHOLD:
            spk_id = best_spk
            self.spk_embs[spk_id].append(emb)
            if self.embedding_update_enabled:
                self.mean_embs[spk_id] = np.median(self.spk_embs[spk_id], axis=0)
            self.curr_spk = spk_id
            return spk_id, best_sim
        elif best_sim >= self.change_thresh:
            spk_id = best_spk
            self.curr_spk = spk_id
            return spk_id, best_sim
        else:
            if self.pending_enabled and len(self.active_spks) < self.max_spks:
                self.pending_embs.append(emb)
                self.pending_times.append(seg_time)
                self._check_pending_promotion()
                return "pending", best_sim
            else:
                spk_id = best_spk
                self.curr_spk = spk_id
                return spk_id, best_sim
    
    def _check_pending_promotion(self):
        if len(self.pending_embs) < MIN_CLUSTER_SIZE or len(self.active_spks) >= self.max_spks:
            if len(self.active_spks) >= self.max_spks:
                self.pending_enabled = False
            return False
        
        cohesive_group = self._find_cohesive_group()
        if cohesive_group is not None:
            start_idx, end_idx = cohesive_group
            new_spk_id = self._get_next_speaker_id()
            if new_spk_id is not None:
                group_embs = self.pending_embs[start_idx:end_idx+1]
                self.spk_embs[new_spk_id] = group_embs
                self.mean_embs[new_spk_id] = np.median(group_embs, axis=0)
                self.active_spks.add(new_spk_id)
                
                promoted_start_time = self.pending_times[start_idx]
                promoted_end_time = self.pending_times[end_idx]
                
                if self.timeline_manager:
                    self.timeline_manager.update_pending_segments_to_speaker(
                        promoted_start_time, promoted_end_time, new_spk_id
                    )
                
                self.pending_embs = []
                self.pending_times = []
                
                if self.embedding_updated:
                    self.embedding_updated()
                return True
        return False
    
    def _find_cohesive_group(self):
        if len(self.pending_embs) < MIN_CLUSTER_SIZE:
            return None
        
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None, distance_threshold=AUTO_CLUSTER_DISTANCE_THRESHOLD,
                metric='cosine', linkage='average'
            )
            labels = clustering.fit_predict(np.array(self.pending_embs))
            
            unique_labels = np.unique(labels)
            cluster_sizes = {label: np.sum(labels == label) for label in unique_labels}
            
            target_cluster = max(cluster_sizes, key=cluster_sizes.get)
            largest_cluster_size = cluster_sizes[target_cluster]
            
            if largest_cluster_size >= MIN_CLUSTER_SIZE:
                target_indices = np.where(labels == target_cluster)[0]
                start_idx = target_indices[0]
                end_idx = target_indices[-1]
                return (start_idx, end_idx)
                
        except Exception as e:
            print(f"Clustering error: {e}")
        return None
    
    def _get_next_speaker_id(self):
        for i in range(self.max_spks):
            if i not in self.active_spks:
                return i
        return None
    
    def get_all_embeddings(self):
        all_embs, labels = [], []
        for spk_id in range(self.max_spks):
            if spk_id in self.active_spks and self.spk_embs[spk_id]:
                for emb in self.spk_embs[spk_id]:
                    all_embs.append(emb)
                    labels.append(spk_id)
        for emb in self.pending_embs:
            all_embs.append(emb)
            labels.append(-1)
        return np.array(all_embs) if all_embs else None, labels
    
    def get_total_embedding_count(self):
        total_count = sum(len(self.spk_embs[spk_id]) for spk_id in range(self.max_spks) if spk_id in self.active_spks)
        return total_count + len(self.pending_embs)
    
    def recluster_spks(self, target_clusters=None):
        all_embs, emb_map = [], []
        for spk_id, embs in enumerate(self.spk_embs):
            for emb in embs:
                all_embs.append(emb)
                emb_map.append(spk_id)
        for emb in self.pending_embs:
            all_embs.append(emb)
            emb_map.append(-1)
        
        if len(all_embs) < 2:
            return False
        
        n_clusters = min(target_clusters or len(self.active_spks), len(all_embs), self.max_spks)
        X = np.array(all_embs)
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
        labels = clustering.fit_predict(X)
        
        self.spk_embs = [[] for _ in range(self.max_spks)]
        self.mean_embs = [None] * self.max_spks
        self.active_spks = set()
        self.pending_embs = []
        self.pending_times = []
        
        for emb, new_label in zip(all_embs, labels):
            if new_label < self.max_spks:
                self.spk_embs[new_label].append(emb)
                self.active_spks.add(new_label)
        
        for i, embs in enumerate(self.spk_embs):
            if embs: 
                self.mean_embs[i] = np.median(embs, axis=0)
        
        if self.embedding_updated:
            self.embedding_updated()
        return True
    
    def toggle_pending(self):
        self.pending_enabled = not self.pending_enabled
        return self.pending_enabled
    
    def toggle_embedding_update(self):
        self.embedding_update_enabled = not self.embedding_update_enabled
        return self.embedding_update_enabled
    
    def reset(self):
        self.curr_spk = None
        self.mean_embs = [None] * self.max_spks
        self.spk_embs = [[] for _ in range(self.max_spks)]
        self.active_spks = set()
        self.pending_embs = []
        self.pending_times = []
        self.pending_enabled = True
        if self.embedding_updated:
            self.embedding_updated()

class AudioCapture(QThread):
    chunk_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, device_name=None, use_mic=False):
        super().__init__()
        self._running = True
        self._paused = False
        self.use_mic = use_mic
        self.device_name = device_name
        self.device = None
        self._setup_device()
    
    def _setup_device(self):
        try:
            if self.device_name:
                self.device = sc.get_microphone(id=self.device_name, include_loopback=not self.use_mic)
            else:
                if self.use_mic:
                    self.device = sc.default_microphone()
                else:
                    self.device = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
        except Exception as e:
            print(f"Error setting up audio device: {e}")
            # Fallback to default
            if self.use_mic:
                self.device = sc.default_microphone()
            else:
                self.device = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
    
    def change_device(self, device_name, use_mic):
        self.device_name = device_name
        self.use_mic = use_mic
        self._setup_device()
    
    def run(self):
        if not self.device:
            print("No audio device available")
            return
            
        try:
            with self.device.recorder(samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE) as recorder:
                while self._running:
                    if self._paused:
                        time.sleep(0.1)
                        continue
                    try:
                        audio_data = recorder.record(numframes=CHUNK_SIZE)
                        if audio_data.size == 0:
                            continue
                        if len(audio_data.shape) > 1:
                            audio_data = audio_data[:, 0]
                        audio_data = audio_data.flatten().astype(np.float32)
                        max_val = np.max(np.abs(audio_data))
                        if max_val > 1.0: 
                            audio_data = audio_data / max_val
                        self.chunk_ready.emit(audio_data)
                    except Exception as e:
                        print(f"Audio recording error: {e}")
                        time.sleep(0.1)
        except Exception as e:
            print(f"Error initializing audio recorder: {e}")
    
    def pause(self):
        self._paused = True
    
    def resume(self):
        self._paused = False
    
    def stop(self): 
        self._running = False


class AudioProcessor(QThread):
    tl_updated = pyqtSignal(list)
    
    def __init__(self, encoder, vad_processor, spk_handler, tl_manager):
        super().__init__()
        self._running = True
        self._paused = False
        self.encoder = encoder
        self.vad_processor = vad_processor
        self.spk_handler = spk_handler
        self.tl_manager = tl_manager
        self.spk_handler.set_timeline_manager(self.tl_manager)
        
        self.buffer = np.zeros(WIN_SAMPLES, dtype=np.float32)
        self.buf_idx = 0
        self.buf_full = False
        self.last_proc_time = 0
        self.pending_vad_tasks = {}
        self.pending_segs = {}
        self.last_ui_update = 0
        self.pending_ui_update = False
    
    def add_chunk(self, audio_data):
        if self._paused:
            return
        self._add_buf(audio_data)
        curr_time = time.time()
        if curr_time - self.last_proc_time >= WINDOW_PROCESS_INTERVAL:
            self.last_proc_time = curr_time
            self._proc_window()
        self._proc_vad_results()
        self._proc_emb_res()
        self._check_ui_update()
    
    def _check_ui_update(self):
        curr_time = time.time()
        if self.pending_ui_update and curr_time - self.last_ui_update >= TIMELINE_UPDATE_INTERVAL:
            self.last_ui_update = curr_time
            self.pending_ui_update = False
            self.tl_updated.emit(self.tl_manager.get_segs())
    
    def _add_buf(self, audio_chunk):
        chunk_len = len(audio_chunk)
        if self.buf_idx + chunk_len <= WIN_SAMPLES:
            self.buffer[self.buf_idx:self.buf_idx + chunk_len] = audio_chunk
            self.buf_idx += chunk_len
        else:
            remaining = WIN_SAMPLES - self.buf_idx
            self.buffer[self.buf_idx:] = audio_chunk[:remaining]
            self.buffer[:chunk_len - remaining] = audio_chunk[remaining:]
            self.buf_idx = chunk_len - remaining
            self.buf_full = True
    
    def _get_window(self):
        if not self.buf_full:
            return self.buffer[:self.buf_idx] if self.buf_idx > 0 else None
        window = np.empty(WIN_SAMPLES, dtype=np.float32)
        window[:WIN_SAMPLES - self.buf_idx] = self.buffer[self.buf_idx:]
        window[WIN_SAMPLES - self.buf_idx:] = self.buffer[:self.buf_idx]
        return window
    
    def _proc_window(self):
        window = self._get_window()
        if window is None or len(window) < SAMPLE_RATE * 0.5:
            return
        timeline_time = self.tl_manager.get_timeline_time()
        seg_start = timeline_time - WINDOW_SIZE
        vad_task_id = self.vad_processor.detect_async(window)
        if vad_task_id:
            seg = Segment(seg_start, WINDOW_SIZE, is_speech=False)
            self.pending_vad_tasks[vad_task_id] = (seg, seg_start, window.copy())
    
    def _proc_vad_results(self):
        while True:
            vad_result = self.vad_processor.get_result()
            if vad_result is None:
                break
            task_id, is_speech = vad_result
            if task_id in self.pending_vad_tasks:
                seg, seg_time, window = self.pending_vad_tasks.pop(task_id)
                seg.is_speech = is_speech
                if is_speech:
                    emb_task_id = self.encoder.embed_async(window)
                    if emb_task_id:
                        self.pending_segs[emb_task_id] = (seg, seg_time)
                else:
                    self.tl_manager.add_seg(seg)
                    self.pending_ui_update = True
    
    def _proc_emb_res(self):
        while True:
            res = self.encoder.get_res()
            if res is None:
                break
            task_id, emb = res
            if task_id in self.pending_segs:
                seg, seg_time = self.pending_segs.pop(task_id)
                if emb is not None:
                    spk_id, sim = self.spk_handler.classify_spk(emb, seg_time)
                    seg.spk_id = spk_id
                    seg.emb = emb
                self.tl_manager.add_seg(seg)
                self.pending_ui_update = True
    
    def pause(self):
        self._paused = True
    
    def resume(self):
        self._paused = False
    
    def reset_tl(self):
        self.spk_handler.reset()
        self.tl_manager.reset()
        self.buffer.fill(0)
        self.buf_idx = 0
        self.buf_full = False
        self.pending_vad_tasks.clear()
        self.pending_segs.clear()
        self.pending_ui_update = False
        self.tl_updated.emit([])
    
    def recluster_spks(self, target_clusters=None):
        if self.spk_handler.recluster_spks(target_clusters):
            self.tl_manager.reclassify_segs(self.spk_handler)
            self.tl_updated.emit(self.tl_manager.get_segs())
            return True
        return False
    
    def stop(self): 
        self._running = False

class Segment:
    def __init__(self, start_time, duration, spk_id=None, emb=None, is_speech=False):
        self.start_time = start_time
        self.duration = duration
        self.spk_id = spk_id
        self.emb = emb
        self.is_speech = is_speech
    
    @property
    def end_time(self): 
        return self.start_time + self.duration

class Timeline:
    def __init__(self):
        self.segs = []
        self.start_time = None
        self.paused_time = None
        self.total_paused_duration = 0
    
    def start_timeline(self):
        if self.start_time is None:
            self.start_time = time.time()
            self.paused_time = None
            self.total_paused_duration = 0
    
    def pause_timeline(self):
        if self.start_time is not None and self.paused_time is None:
            self.paused_time = time.time()
    
    def resume_timeline(self):
        if self.paused_time is not None:
            self.total_paused_duration += time.time() - self.paused_time
            self.paused_time = None
    
    def get_timeline_time(self):
        if self.start_time is None:
            return 0
        current_time = time.time()
        if self.paused_time is not None:
            return self.paused_time - self.start_time - self.total_paused_duration
        else:
            return current_time - self.start_time - self.total_paused_duration
    
    def add_seg(self, seg):
        if self.start_time is not None and self.paused_time is None:
            self.segs.append(seg)
    
    def update_pending_segments_to_speaker(self, start_time, end_time, new_speaker_id):
        for seg in self.segs:
            if (seg.spk_id == "pending" and seg.start_time >= start_time and seg.start_time <= end_time):
                seg.spk_id = new_speaker_id
    
    def get_segs(self):
        return self.segs
    
    def get_timeline_duration(self):
        return self.get_timeline_time()
    
    def reset(self):
        self.segs = []
        self.start_time = None
        self.paused_time = None
        self.total_paused_duration = 0
    
    def reclassify_segs(self, spk_handler):
        for seg in self.segs:
            if seg.is_speech and seg.emb is not None:
                best_spk, best_sim = None, -1.0
                for i, mean_emb in enumerate(spk_handler.mean_embs):
                    if mean_emb is not None:
                        sim = 1.0 - cosine(seg.emb, mean_emb)
                        if sim > best_sim:
                            best_sim = sim
                            best_spk = i
                seg.spk_id = best_spk

class EmbeddingVisualizationWindow(QMainWindow):
    def __init__(self, spk_handler):
        super().__init__()
        self.spk_handler = spk_handler
        self.setWindowTitle("Speaker Embedding Visualization")
        self.setGeometry(100, 100, 800, 600)
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.setCentralWidget(self.canvas)
        self.pca = PCA(n_components=2)
        self.update_plot()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(2000)
    
    def update_plot(self):
        embeddings, labels = self.spk_handler.get_all_embeddings()
        if embeddings is None or len(embeddings) < 2:
            return
        
        embeddings_2d = self.pca.fit_transform(embeddings)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        unique_labels = set(labels)
        
        for label in unique_labels:
            mask = np.array(labels) == label
            points = embeddings_2d[mask]
            if label == -1:
                ax.scatter(points[:, 0], points[:, 1], c=PENDING_COLOR, alpha=0.7, s=50, 
                          label=f"Pending ({len(points)})")
            else:
                color = SPEAKER_COLORS[label % len(SPEAKER_COLORS)]
                ax.scatter(points[:, 0], points[:, 1], c=color, alpha=0.7, s=50, 
                          label=f"Speaker {label+1} ({len(points)})")
        
        mean_embeddings, mean_labels = [], []
        for spk_id in self.spk_handler.active_spks:
            if self.spk_handler.mean_embs[spk_id] is not None:
                mean_embeddings.append(self.spk_handler.mean_embs[spk_id])
                mean_labels.append(spk_id)
        
        if mean_embeddings:
            mean_embeddings_2d = self.pca.transform(np.array(mean_embeddings))
            for mean_point, spk_id in zip(mean_embeddings_2d, mean_labels):
                color = SPEAKER_COLORS[spk_id % len(SPEAKER_COLORS)]
                ax.scatter(mean_point[0], mean_point[1], c=color, marker='*', s=200, 
                          edgecolors='black', linewidth=1)
        
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_title('Speaker Embeddings Visualization (PCA)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        stats_text = f"Total: {len(embeddings)}, Active: {len(self.spk_handler.active_spks)}, " \
                    f"Pending: {len(self.spk_handler.pending_embs)}\n" \
                    f"Pending: {self.spk_handler.pending_enabled}, Update: {self.spk_handler.embedding_update_enabled}\n" \
                    f"Min cluster size: {MIN_CLUSTER_SIZE}, Embedding update thresh: {EMBEDDING_UPDATE_THRESHOLD}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        self.canvas.draw()

class TimelineUI(QWidget):
    def __init__(self):
        super().__init__()
        self.segs = []
        self.max_spks = MAX_SPEAKERS
        self.pixels_per_second = 100
        self.setMinimumHeight(TIMELINE_HEIGHT)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor("#1A1A1A"))
        self.setPalette(palette)
        self.last_size_update = 0
        self.current_width = 800
    
    def update_segs(self, segs): 
        self.segs = segs
        self._update_size_throttled()
        self.update()
    
    def _update_size_throttled(self):
        curr_time = time.time()
        if curr_time - self.last_size_update >= SIZE_UPDATE_INTERVAL:
            self.last_size_update = curr_time
            self._update_size()
    
    def _update_size(self):
        if not self.segs:
            new_width = 800
        else:
            max_time = max(seg.end_time for seg in self.segs)
            new_width = max(800, int((max_time + 5) * self.pixels_per_second))
        if abs(new_width - self.current_width) > 50:
            self.current_width = new_width
            self.setMinimumWidth(new_width)
    
    def _format_time(self, seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        painter.fillRect(0, 0, width, height, QBrush(QColor("#1A1A1A")))
        
        if not self.segs: 
            return
        
        total_layers = self.max_spks + 2
        layer_h = max(40, height // total_layers)
        max_time = max(seg.end_time for seg in self.segs) if self.segs else 0
        
        painter.setPen(QPen(QColor("#444444"), 1))
        for i in range(1, total_layers):
            y = int(i * layer_h)
            painter.drawLine(0, y, width, y)
        
        painter.setPen(QPen(QColor("#666666")))
        painter.setFont(QFont("Arial", 10))
        for i in range(0, int(max_time) + 10, 10):
            x = int(i * self.pixels_per_second)
            if x <= width:
                painter.drawLine(x, 0, x, height)
                time_str = self._format_time(i)
                painter.drawText(x + 5, 15, time_str)
        
        visible_start = max(0, event.rect().left() / self.pixels_per_second - 1)
        visible_end = (event.rect().right() / self.pixels_per_second) + 1
        
        for seg in self.segs:
            if seg.end_time < visible_start or seg.start_time > visible_end:
                continue
            x_start = int(seg.start_time * self.pixels_per_second)
            x_end = int(seg.end_time * self.pixels_per_second)
            if x_start > width or x_end < 0: 
                continue
            x_start, x_end = max(0, x_start), min(width, x_end)
            
            if seg.is_speech and seg.spk_id is not None:
                if seg.spk_id == "pending":
                    layer = self.max_spks
                    color = QColor(PENDING_COLOR)
                else:
                    layer = seg.spk_id % self.max_spks
                    color = QColor(SPEAKER_COLORS[seg.spk_id % len(SPEAKER_COLORS)])
            else:
                layer = self.max_spks + 1
                color = QColor("#666666")
            
            color.setAlpha(120)
            y = int(layer * layer_h + 5)
            rect_h = layer_h - 10
            painter.fillRect(x_start, y, x_end - x_start, rect_h, QBrush(color))
        
        painter.setPen(QPen(QColor("#CCCCCC")))
        painter.setFont(QFont("Arial", 12))
        for i in range(self.max_spks):
            y = int(i * layer_h + layer_h // 2 + 5)
            painter.drawText(10, y, f"Speaker {i+1}")
        
        y = int(self.max_spks * layer_h + layer_h // 2 + 5)
        painter.drawText(10, y, "Pending")
        y = int((self.max_spks + 1) * layer_h + layer_h // 2 + 5)
        painter.drawText(10, y, "Non-Speech")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Speaker Timeline with Auto-Clustering (ECAPA-TDNN + Silero VAD)")
        self.encoder = None
        self.vad_processor = None
        self.audio_capture = None
        self.audio_proc = None
        self.spk_handler = SpeakerHandler()
        self.tl_manager = Timeline()
        self.is_recording = False
        self.visualization_window = None
        self.models_loaded = {"encoder": False, "vad": False}
        self.spk_handler.set_embedding_callback(self._on_embeddings_updated)
        self._setup_ui()
        QTimer.singleShot(500, self._init_app)
    
    def _get_audio_devices(self):
        """Get list of available audio devices"""
        try:
            microphones = [(mic.name, True) for mic in sc.all_microphones()]
            speakers = [(spk.name, False) for spk in sc.all_speakers()]
            return microphones + speakers
        except Exception as e:
            print(f"Error getting audio devices: {e}")
            return []
    
    def _refresh_devices(self):
        """Refresh the audio device list"""
        self.device_combo.clear()
        devices = self._get_audio_devices()
        
        for device_name, is_mic in devices:
            device_type = "Microphone" if is_mic else "Speaker"
            display_name = f"[{device_type}] {device_name}"
            self.device_combo.addItem(display_name, (device_name, is_mic))
        
        # Set default selection
        if self.device_combo.count() > 0:
            # Try to find default speaker first (for loopback)
            try:
                default_speaker_name = str(sc.default_speaker().name)
                for i in range(self.device_combo.count()):
                    device_name, is_mic = self.device_combo.itemData(i)
                    if device_name == default_speaker_name and not is_mic:
                        self.device_combo.setCurrentIndex(i)
                        break
                else:
                    self.device_combo.setCurrentIndex(0)
            except:
                self.device_combo.setCurrentIndex(0)
    
    def _apply_device_change(self):
        """Apply the selected audio device"""
        if self.device_combo.currentIndex() < 0:
            return
        
        device_name, is_mic = self.device_combo.currentData()
        
        # Stop current recording if active
        was_recording = self.is_recording
        if was_recording:
            self._toggle_recording()  # Pause
        
        # Stop and recreate audio capture
        if self.audio_capture:
            self.audio_capture.stop()
            self.audio_capture.wait()
        
        # Create new audio capture with selected device
        self.audio_capture = AudioCapture(device_name=device_name, use_mic=is_mic)
        self.audio_capture.pause()
        self.audio_capture.chunk_ready.connect(self.audio_proc.add_chunk)
        self.audio_capture.start()
        
        # Update status
        device_type = "Microphone" if is_mic else "Speaker"
        self.status_label.setText(f"Audio device changed to: [{device_type}] {device_name}")
        
        # Resume recording if it was active
        if was_recording:
            QTimer.singleShot(100, self._toggle_recording)  # Resume after a short delay
    
    def _on_embeddings_updated(self):
        if self.visualization_window and self.visualization_window.isVisible():
            self.visualization_window.update_plot()
    
    def _setup_ui(self):
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)
        
        self.status_label = QLabel("Preparing...")
        layout.addWidget(self.status_label)
        
        self.scroll_area = QScrollArea()
        self.timeline = TimelineUI()
        self.scroll_area.setWidget(self.timeline)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(self.scroll_area, 1)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        self.start_pause_btn = QPushButton("Start")
        self.start_pause_btn.clicked.connect(self._toggle_recording)
        self.start_pause_btn.setEnabled(False)
        btn_layout.addWidget(self.start_pause_btn)
        
        self.reset_btn = QPushButton("Reset Timeline")
        self.reset_btn.clicked.connect(self._reset_tl)
        self.reset_btn.setEnabled(False)
        btn_layout.addWidget(self.reset_btn)
        
        self.recluster_btn = QPushButton("Recluster Speakers")
        self.recluster_btn.clicked.connect(self._recluster)
        self.recluster_btn.setEnabled(False)
        btn_layout.addWidget(self.recluster_btn)
        
        self.viz_btn = QPushButton("Embedding Visualization")
        self.viz_btn.clicked.connect(self._show_visualization)
        self.viz_btn.setEnabled(False)
        btn_layout.addWidget(self.viz_btn)
        
        self.pending_btn = QPushButton("Disable Pending")
        self.pending_btn.clicked.connect(self._toggle_pending)
        self.pending_btn.setEnabled(False)
        btn_layout.addWidget(self.pending_btn)
        
        self.embedding_update_btn = QPushButton("Disable Embedding Update")
        self.embedding_update_btn.clicked.connect(self._toggle_embedding_update)
        self.embedding_update_btn.setEnabled(False)
        btn_layout.addWidget(self.embedding_update_btn)
        
        layout.addLayout(btn_layout)
        
        # Audio device selection
        device_group = QGroupBox("Audio Device Selection")
        device_layout = QHBoxLayout(device_group)
        
        device_layout.addWidget(QLabel("Device:"))
        
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(300)
        device_layout.addWidget(self.device_combo)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        device_layout.addWidget(self.refresh_btn)
        
        self.apply_device_btn = QPushButton("Apply")
        self.apply_device_btn.clicked.connect(self._apply_device_change)
        self.apply_device_btn.setEnabled(False)
        device_layout.addWidget(self.apply_device_btn)
        
        device_layout.addStretch()
        layout.addWidget(device_group)
        
        # Initialize device list
        self._refresh_devices()
        
        self.scroll_timer = QTimer()
        self.scroll_timer.timeout.connect(self._auto_scroll)
        self.scroll_timer.start(2000)
        
        self.setStyleSheet("""
            QMainWindow, QWidget {background-color: #2D2D30; color: #CCCCCC;}
            QPushButton {background: #3F3F46; color: #EEEEEE; border: 1px solid #555555; 
                        padding: 8px 15px; margin: 5px;}
            QPushButton:hover {background: #505059;}
            QPushButton:disabled {background: #333337; color: #777777;}
            QLabel {padding: 5px; font-size: 14px;}
            QScrollArea {border: 1px solid #555555;}
            QGroupBox {font-weight: bold; border: 2px solid #555555; margin: 10px 0px; padding-top: 10px;}
            QGroupBox::title {subcontrol-origin: margin; left: 10px; padding: 0px 5px 0px 5px;}
            QComboBox {background: #3F3F46; color: #EEEEEE; border: 1px solid #555555; 
                      padding: 5px 10px; margin: 5px;}
            QComboBox::drop-down {border: none;}
            QComboBox::down-arrow {image: none; border: none;}
        """)
    
    def _auto_scroll(self):
        if self.is_recording:
            scrollbar = self.scroll_area.horizontalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def _toggle_recording(self):
        if not self.is_recording:
            self.tl_manager.start_timeline()
            self.tl_manager.resume_timeline()
            if self.audio_capture:
                self.audio_capture.resume()
            if self.audio_proc:
                self.audio_proc.resume()
            self.is_recording = True
            self.start_pause_btn.setText("Pause")
            self.status_label.setText("Recording...")
        else:
            self.tl_manager.pause_timeline()
            if self.audio_capture:
                self.audio_capture.pause()
            if self.audio_proc:
                self.audio_proc.pause()
            self.is_recording = False
            self.start_pause_btn.setText("Resume")
            self.status_label.setText("Paused")
    
    def _reset_tl(self):
        if self.audio_proc:
            self.audio_proc.reset_tl()
        self.is_recording = False
        self.start_pause_btn.setText("Start")
        self.status_label.setText("Timeline has been reset.")
    
    def _recluster(self):
        if not self.audio_proc:
            return
        total_embeddings = self.spk_handler.get_total_embedding_count()
        if total_embeddings < 50:
            QMessageBox.information(
                self, "Information", 
                f"Not enough embeddings for reclustering.\n"
                f"Current embeddings: {total_embeddings}\n"
                f"Required embeddings: 50 or more"
            )
            return
        
        active_speakers = len(self.spk_handler.active_spks)
        cluster_count, ok = QInputDialog.getInt(
            self, "Recluster Speakers", 
            f"Enter number of speakers:\n"
            f"Total embeddings: {total_embeddings}\n"
            f"Currently detected speakers: {active_speakers}:",
            value=max(2, active_speakers), min=2, max=MAX_SPEAKERS
        )
        
        if ok:
            success = self.audio_proc.recluster_spks(cluster_count)
            if success:
                self.status_label.setText(f"Reclustered {total_embeddings} embeddings into {cluster_count} speakers.")
            else:
                QMessageBox.warning(self, "Error", "Reclustering failed.")
    
    def _toggle_pending(self):
        if not self.spk_handler:
            return
        is_enabled = self.spk_handler.toggle_pending()
        if is_enabled:
            self.pending_btn.setText("Disable Pending")
            self.status_label.setText("Pending feature enabled.")
        else:
            self.pending_btn.setText("Enable Pending")
            self.status_label.setText("Pending feature disabled.")
    
    def _toggle_embedding_update(self):
        if not self.spk_handler:
            return
        is_enabled = self.spk_handler.toggle_embedding_update()
        if is_enabled:
            self.embedding_update_btn.setText("Disable Embedding Update")
            self.status_label.setText("Embedding update enabled.")
        else:
            self.embedding_update_btn.setText("Enable Embedding Update")
            self.status_label.setText("Embedding update disabled.")
    
    def _show_visualization(self):
        if self.visualization_window is None:
            self.visualization_window = EmbeddingVisualizationWindow(self.spk_handler)
        self.visualization_window.show()
        self.visualization_window.raise_()
        self.visualization_window.activateWindow()
    
    def _init_app(self):
        self.resize(1400, 900)  # Increased height for device selection area
        device = "cuda" if DEVICE_PREF == "cuda" and torch.cuda.is_available() else "cpu"
        if DEVICE_PREF == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
        else:
            print(f"Using {device.upper()} device")
        
        self.encoder = SpeechBrainEncoder(device)
        self.encoder.model_loaded.connect(self._on_encoder_loaded)
        self.encoder.start()
        
        self.vad_processor = SileroVAD(device, VAD_THRESH)
        self.vad_processor.model_loaded.connect(self._on_vad_loaded)
        self.vad_processor.start()
        
        self.status_label.setText("Loading models... (Encoder + VAD)")
    
    def _on_encoder_loaded(self):
        self.models_loaded["encoder"] = True
        self._check_models_ready()
    
    def _on_vad_loaded(self):
        self.models_loaded["vad"] = True
        self._check_models_ready()
    
    def _check_models_ready(self):
        if all(self.models_loaded.values()):
            # Get initial device selection
            device_name, use_mic = None, False
            if self.device_combo.currentIndex() >= 0:
                device_name, use_mic = self.device_combo.currentData()
            
            self.audio_capture = AudioCapture(device_name=device_name, use_mic=use_mic)
            self.audio_proc = AudioProcessor(self.encoder, self.vad_processor, 
                                           self.spk_handler, self.tl_manager)
            
            self.audio_capture.pause()
            self.audio_proc.pause()
            
            self.audio_capture.chunk_ready.connect(self.audio_proc.add_chunk)
            self.audio_proc.tl_updated.connect(self.timeline.update_segs)
            
            self.audio_proc.start()
            self.audio_capture.start()
            
            self.start_pause_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)
            self.recluster_btn.setEnabled(True)
            self.viz_btn.setEnabled(True)
            self.pending_btn.setEnabled(True)
            self.embedding_update_btn.setEnabled(True)
            self.apply_device_btn.setEnabled(True)
            
            # Show current device in status
            if device_name:
                device_type = "Microphone" if use_mic else "Speaker"
                self.status_label.setText(f"Ready - Current device: [{device_type}] {device_name}")
            else:
                self.status_label.setText("Ready - Press Start button to begin")
    
    def closeEvent(self, event):
        if self.visualization_window:
            self.visualization_window.close()
        if self.audio_capture: 
            self.audio_capture.stop()
        if self.audio_proc: 
            self.audio_proc.stop()
        if self.encoder:
            self.encoder.stop_proc()
        if self.vad_processor:
            self.vad_processor.stop_processing()
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
