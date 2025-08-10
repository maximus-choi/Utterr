import soundcard as sc, numpy as np, threading, time, sys, os, urllib.request, queue, torch, torchaudio, requests, json
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen, QFont, QTextCharFormat, QTextCursor
from scipy.spatial.distance import cosine
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
import azure.cognitiveservices.speech as speechsdk
from concurrent.futures import ThreadPoolExecutor

SPEAKER_COLORS = ["#FF4444", "#44FF44", "#4444FF", "#FFFF44", "#FF44FF", "#44FFFF", "#FF8844", "#FF009D", "#8844FF", "#FFAA44"]
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
PENDING_THRESHOLD = 0.4
EMBEDDING_UPDATE_THRESHOLD = 0.5
MIN_PENDING_SIZE = 15
AUTO_CLUSTER_DISTANCE_THRESHOLD = 0.6
MIN_CLUSTER_SIZE = 10
SPEECH_KEY = ""
SERVICE_REGION = "koreacentral"
SPEECH_LANGUAGE = "ko-KR"
TRANSLATION_LANGUAGE = "en"

def translate_text(text, source_lang='ko', target_lang='en'):
    try:
        response = requests.get('https://translate.googleapis.com/translate_a/single',
            params={'client': 'gtx', 'sl': source_lang, 'tl': target_lang, 'dt': 't', 'q': text})
        if response.status_code == 200:
            result = json.loads(response.text)
            return ''.join([sentence[0] for sentence in result[0]])
        return "Translation failed"
    except Exception as e:
        print(f"Translation error: {e}")
        return text

class SileroVAD(QThread):
    model_loaded = pyqtSignal()
    
    def __init__(self, device="cpu", threshold=0.5):
        super().__init__()
        self.device, self.threshold = "cpu", threshold
        self.vad_model = self.get_speech_ts = None
        self.model_loaded_flag = False
        self.vad_queue, self.res_queue = queue.Queue(), queue.Queue()
        self._stop_proc = False
    
    def run(self):
        try:
            print("Loading Silero VAD model on CPU...")
            model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False, onnx=False)
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
            except queue.Empty: continue
            except Exception as e: print(f"VAD processing error: {e}")
    
    def _detect_speech(self, audio_data, sr=16000):
        if not self.model_loaded_flag or self.vad_model is None or len(audio_data) < 1600: return False
        try:
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
            with torch.no_grad():
                speech_timestamps = self.get_speech_ts(audio_tensor, self.vad_model, threshold=self.threshold, 
                    sampling_rate=sr, return_seconds=False)
            return len(speech_timestamps) > 0
        except Exception as e:
            print(f"Speech detection error: {e}")
            return False
    
    def detect_async(self, audio_data, sr=16000):
        if not self.model_loaded_flag: return None
        task_id = time.time()
        self.vad_queue.put((task_id, audio_data.copy(), sr))
        return task_id
    
    def get_result(self):
        try: return self.res_queue.get_nowait()
        except queue.Empty: return None
    
    def stop_processing(self): self._stop_proc = True

class SpeechBrainEncoder(QThread):
    model_loaded = pyqtSignal()
    
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.model = None
        self.model_loaded_flag = False
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "speechbrain")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.emb_queue, self.res_queue = queue.Queue(), queue.Queue()
        self._stop_proc = False
    
    def run(self):
        model_url = "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.ckpt"
        model_path = os.path.join(self.cache_dir, "embedding_model_ecapa.ckpt")
        if not os.path.exists(model_path):
            print(f"Downloading ECAPA-TDNN model to {model_path}...")
            urllib.request.urlretrieve(model_url, model_path)
        from speechbrain.pretrained import EncoderClassifier
        self.model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir=self.cache_dir, run_opts={"device": self.device})
        self.model_loaded_flag = True
        self.model_loaded.emit()
        while not self._stop_proc:
            try:
                task_id, audio, sr = self.emb_queue.get(timeout=0.1)
                emb = self._compute_emb(audio, sr)
                self.res_queue.put((task_id, emb))
            except queue.Empty: continue
            except Exception as e: print(f"Embedding error: {e}")
    
    def _compute_emb(self, audio, sr=16000):
        if not self.model_loaded_flag: return None
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad(): emb = self.model.encode_batch(waveform)
        return emb.squeeze().cpu().numpy()
    
    def embed_async(self, audio, sr=16000):
        if not self.model_loaded_flag: return None
        task_id = time.time()
        self.emb_queue.put((task_id, audio.copy(), sr))
        return task_id
    
    def get_res(self):
        try: return self.res_queue.get_nowait()
        except queue.Empty: return None
    
    def stop_proc(self): self._stop_proc = True

class STTWorker(QThread):
    update_text = pyqtSignal(str, str, object)
    
    def __init__(self, timeline_manager):
        super().__init__()
        self.timeline_manager = timeline_manager
        self.speech_language, self.translation_language = SPEECH_LANGUAGE, TRANSLATION_LANGUAGE
        self._init_azure_config()
        self._setup_audio_stream()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = True
        self.lock = threading.Lock()
        self.current_transcript = [""]
        self.current_speaker = None
        self.stt_start_time = self.timeline_start_time = None
        self.time_offset = 0.0

    def _init_azure_config(self):
        self.speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SERVICE_REGION)
        self.speech_config.request_word_level_timestamps()
        self.speech_config.output_format = speechsdk.OutputFormat.Detailed
        self.speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_StablePartialResultThreshold, "3")
        self.speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "500")

    def _setup_audio_stream(self):
        self.audio_format = speechsdk.audio.AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1)
        self.audio_stream = speechsdk.audio.PushAudioInputStream(self.audio_format)
        self.audio_config = speechsdk.audio.AudioConfig(stream=self.audio_stream)

    def start_stt_with_timeline_sync(self):
        self.timeline_start_time = self.timeline_manager.start_time
        self.stt_start_time = time.time()
        if self.timeline_start_time: self.time_offset = self.stt_start_time - self.timeline_start_time
        print(f"STT-Timeline sync: offset = {self.time_offset:.3f}s")

    def _format_time(self, ticks): return ticks / 10_000_000

    def _adjust_timestamp_to_timeline(self, stt_timestamp): return max(0, stt_timestamp - self.time_offset)

    def _extract_word_timestamps(self, result):
        word_timestamps = []
        try:
            detailed_result = json.loads(result.json)
            if "NBest" in detailed_result and detailed_result["NBest"]:
                best_result = detailed_result["NBest"][0]
                if "Words" in best_result:
                    for word_info in best_result["Words"]:
                        word = word_info.get("Word", "")
                        offset = word_info.get("Offset", 0)
                        duration = word_info.get("Duration", 0)
                        stt_start_time = self._format_time(offset)
                        stt_end_time = self._format_time(offset + duration)
                        adjusted_start = self._adjust_timestamp_to_timeline(stt_start_time)
                        adjusted_end = self._adjust_timestamp_to_timeline(stt_end_time)
                        word_timestamps.append({'word': word, 'start_time': adjusted_start, 
                            'end_time': adjusted_end, 'duration': adjusted_end - adjusted_start})
                        if len(word_timestamps) == 1:
                            print(f"Timestamp sync - Offset: {self.time_offset:.3f}s, First word '{word}': STT({stt_start_time:.2f}s) → Timeline({adjusted_start:.2f}s)")
        except Exception as e: print(f"타임스탬프 파싱 오류: {str(e)}")
        return word_timestamps

    def set_current_speaker(self, speaker_id): self.current_speaker = speaker_id

    def process_audio(self, chunk):
        try: self.audio_stream.write(chunk)
        except Exception as e: print(f"스트리밍 오류: {str(e)}")

    def run(self):
        self.recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, 
            audio_config=self.audio_config, language=self.speech_language)

        def recognizing_callback(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
                with self.lock:
                    self.current_transcript[0] = evt.result.text
                    self.update_text.emit("recognizing", evt.result.text, None)

        def recognized_callback(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                with self.lock:
                    final_text = evt.result.text
                    if final_text.strip():
                        print(f"최종 결과: {final_text}")
                        word_timestamps = self._extract_word_timestamps(evt.result)
                        source_lang = self.speech_language.split('-')[0]
                        translated = translate_text(final_text, source_lang=source_lang, target_lang=self.translation_language)
                        print(f"번역 결과: {translated}\n")
                        self.update_text.emit("recognized", f"{final_text}\n{translated}", word_timestamps)
                        self.current_transcript[0] = ""

        self.recognizer.recognizing.connect(recognizing_callback)
        self.recognizer.recognized.connect(recognized_callback)
        self.recognizer.start_continuous_recognition()
        print(f"\n=== STT 설정 정보 ===\n음성 인식 언어: {self.speech_language}\n번역 언어: {self.translation_language}\n단어별 타임스탬프: 활성화\n===================\n")

    def stop(self):
        self.running = False
        if hasattr(self, 'recognizer'): self.recognizer.stop_continuous_recognition_async()
        self.executor.shutdown(wait=False)

class SpeakerHandler:
    def __init__(self, max_spks=MAX_SPEAKERS, change_thresh=PENDING_THRESHOLD, min_pending=MIN_PENDING_SIZE):
        self.max_spks, self.change_thresh, self.min_pending = max_spks, change_thresh, min_pending
        self.curr_spk = None
        self.mean_embs = [None] * max_spks
        self.spk_embs = [[] for _ in range(max_spks)]
        self.active_spks = set()
        self.pending_embs, self.pending_times = [], []
        self.pending_enabled = self.embedding_update_enabled = True
        self.embedding_updated = self.timeline_manager = self.speaker_changed_callback = None
        # Pre-recorded speaker management
        self.fixed_speakers = set()  # Set of speaker IDs that are fixed
        self.speaker_names = {}  # Map of speaker ID to custom name
    
    def set_embedding_callback(self, callback): self.embedding_updated = callback
    def set_speaker_changed_callback(self, callback): self.speaker_changed_callback = callback
    def set_timeline_manager(self, timeline_manager): self.timeline_manager = timeline_manager
    
    def register_speaker(self, speaker_id, embeddings, name=None):
        """Register pre-recorded embeddings for a specific speaker"""
        if speaker_id >= self.max_spks: return False
        self.spk_embs[speaker_id] = embeddings
        self.mean_embs[speaker_id] = np.median(embeddings, axis=0)
        self.active_spks.add(speaker_id)
        self.fixed_speakers.add(speaker_id)
        if name: self.speaker_names[speaker_id] = name
        if self.embedding_updated: self.embedding_updated()
        return True
    
    def unfix_speaker(self, speaker_id):
        """Allow a speaker's embeddings to be updated"""
        if speaker_id in self.fixed_speakers:
            self.fixed_speakers.remove(speaker_id)
            return True
        return False
    
    def fix_speaker(self, speaker_id):
        """Fix a speaker's embeddings so they won't be updated"""
        if speaker_id in self.active_spks:
            self.fixed_speakers.add(speaker_id)
            return True
        return False
    
    def remove_speaker(self, speaker_id):
        """Remove a speaker completely"""
        if speaker_id in self.active_spks:
            self.spk_embs[speaker_id] = []
            self.mean_embs[speaker_id] = None
            self.active_spks.remove(speaker_id)
            self.fixed_speakers.discard(speaker_id)
            self.speaker_names.pop(speaker_id, None)
            if self.embedding_updated: self.embedding_updated()
            return True
        return False
    
    def get_speaker_name(self, speaker_id):
        """Get custom name for a speaker"""
        return self.speaker_names.get(speaker_id, f"Speaker {speaker_id + 1}")
    
    def classify_spk(self, emb, seg_time):
        previous_speaker = self.curr_spk
        if not self.active_spks and self.pending_enabled:
            self.pending_embs.append(emb)
            self.pending_times.append(seg_time)
            self._check_pending_promotion()
            result = "pending", 0.0
        elif not self.active_spks:
            self.spk_embs[0].append(emb)
            self.mean_embs[0] = emb
            self.active_spks.add(0)
            self.curr_spk = 0
            result = 0, 1.0
        else:
            active_mean_embs, active_spk_ids = [], []
            for spk_id in self.active_spks:
                if self.mean_embs[spk_id] is not None:
                    active_mean_embs.append(self.mean_embs[spk_id])
                    active_spk_ids.append(spk_id)
            if not active_mean_embs:
                self.spk_embs[0].append(emb)
                self.mean_embs[0] = emb
                self.active_spks.add(0)
                self.curr_spk = 0
                result = 0, 1.0
            else:
                emb_norm = emb / np.linalg.norm(emb)
                mean_embs_matrix = np.array(active_mean_embs)
                mean_embs_norm = mean_embs_matrix / np.linalg.norm(mean_embs_matrix, axis=1, keepdims=True)
                similarities = np.dot(mean_embs_norm, emb_norm)
                best_idx = np.argmax(similarities)
                best_sim = similarities[best_idx]
                best_spk = active_spk_ids[best_idx]
                if best_sim >= EMBEDDING_UPDATE_THRESHOLD:
                    spk_id = best_spk
                    # Only update embeddings if speaker is not fixed
                    if spk_id not in self.fixed_speakers:
                        self.spk_embs[spk_id].append(emb)
                        if self.embedding_update_enabled: 
                            self.mean_embs[spk_id] = np.median(self.spk_embs[spk_id], axis=0)
                    self.curr_spk = spk_id
                    result = spk_id, best_sim
                elif best_sim >= self.change_thresh:
                    self.curr_spk = spk_id = best_spk
                    result = spk_id, best_sim
                else:
                    if self.pending_enabled and len(self.active_spks) < self.max_spks:
                        self.pending_embs.append(emb)
                        self.pending_times.append(seg_time)
                        self._check_pending_promotion()
                        result = "pending", best_sim
                    else:
                        self.curr_spk = spk_id = best_spk
                        result = spk_id, best_sim
        if self.curr_spk != previous_speaker and self.speaker_changed_callback:
            self.speaker_changed_callback(self.curr_spk)
        return result
    
    def _check_pending_promotion(self):
        if len(self.pending_embs) < MIN_CLUSTER_SIZE or len(self.active_spks) >= self.max_spks:
            if len(self.active_spks) >= self.max_spks: self.pending_enabled = False
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
                    self.timeline_manager.update_pending_segments_to_speaker(promoted_start_time, promoted_end_time, new_spk_id)
                self.pending_embs, self.pending_times = [], []
                if self.embedding_updated: self.embedding_updated()
                return True
        return False
    
    def _find_cohesive_group(self):
        if len(self.pending_embs) < MIN_CLUSTER_SIZE: return None
        try:
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=AUTO_CLUSTER_DISTANCE_THRESHOLD,
                metric='cosine', linkage='average')
            labels = clustering.fit_predict(np.array(self.pending_embs))
            unique_labels = np.unique(labels)
            cluster_sizes = {label: np.sum(labels == label) for label in unique_labels}
            target_cluster = max(cluster_sizes, key=cluster_sizes.get)
            largest_cluster_size = cluster_sizes[target_cluster]
            if largest_cluster_size >= MIN_CLUSTER_SIZE:
                target_indices = np.where(labels == target_cluster)[0]
                return (target_indices[0], target_indices[-1])
        except Exception as e: print(f"Clustering error: {e}")
        return None
    
    def _get_next_speaker_id(self):
        for i in range(self.max_spks):
            if i not in self.active_spks: return i
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
        return sum(len(self.spk_embs[spk_id]) for spk_id in range(self.max_spks) if spk_id in self.active_spks) + len(self.pending_embs)
    
    def recluster_spks(self, target_clusters=None):
        # Don't recluster fixed speakers
        all_embs, emb_map = [], []
        fixed_embs = {spk_id: self.spk_embs[spk_id] for spk_id in self.fixed_speakers}
        
        for spk_id, embs in enumerate(self.spk_embs):
            if spk_id not in self.fixed_speakers:
                for emb in embs:
                    all_embs.append(emb)
                    emb_map.append(spk_id)
        for emb in self.pending_embs:
            all_embs.append(emb)
            emb_map.append(-1)
        if len(all_embs) < 2: return False
        
        # Calculate number of clusters excluding fixed speakers
        n_fixed = len(self.fixed_speakers)
        n_clusters = min(target_clusters - n_fixed if target_clusters else len(self.active_spks) - n_fixed, 
                        len(all_embs), self.max_spks - n_fixed)
        
        if n_clusters <= 0: return False
        
        X = np.array(all_embs)
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
        labels = clustering.fit_predict(X)
        
        # Clear non-fixed speakers
        for spk_id in range(self.max_spks):
            if spk_id not in self.fixed_speakers:
                self.spk_embs[spk_id] = []
                self.mean_embs[spk_id] = None
                self.active_spks.discard(spk_id)
        
        self.pending_embs, self.pending_times = [], []
        
        # Restore fixed speakers
        for spk_id, embs in fixed_embs.items():
            self.spk_embs[spk_id] = embs
            self.mean_embs[spk_id] = np.median(embs, axis=0)
            self.active_spks.add(spk_id)
        
        # Find available speaker IDs for new clusters
        available_ids = [i for i in range(self.max_spks) if i not in self.fixed_speakers]
        
        # Assign new clusters to available speaker IDs
        for emb, new_label in zip(all_embs, labels):
            if new_label < len(available_ids):
                spk_id = available_ids[new_label]
                self.spk_embs[spk_id].append(emb)
                self.active_spks.add(spk_id)
        
        # Update mean embeddings for new clusters
        for i in available_ids:
            if self.spk_embs[i]:
                self.mean_embs[i] = np.median(self.spk_embs[i], axis=0)
        
        if self.embedding_updated: self.embedding_updated()
        return True
    
    def toggle_pending(self):
        self.pending_enabled = not self.pending_enabled
        return self.pending_enabled
    
    def toggle_embedding_update(self):
        self.embedding_update_enabled = not self.embedding_update_enabled
        return self.embedding_update_enabled
    
    def reset(self):
        # Keep fixed speakers during reset
        fixed_data = {}
        for spk_id in self.fixed_speakers:
            fixed_data[spk_id] = {
                'embs': self.spk_embs[spk_id].copy(),
                'mean': self.mean_embs[spk_id].copy(),
                'name': self.speaker_names.get(spk_id)
            }
        
        self.curr_spk = None
        self.mean_embs = [None] * self.max_spks
        self.spk_embs = [[] for _ in range(self.max_spks)]
        self.active_spks = set()
        self.pending_embs, self.pending_times = [], []
        self.pending_enabled = True
        
        # Restore fixed speakers
        for spk_id, data in fixed_data.items():
            self.spk_embs[spk_id] = data['embs']
            self.mean_embs[spk_id] = data['mean']
            self.active_spks.add(spk_id)
            if data['name']:
                self.speaker_names[spk_id] = data['name']
        
        if self.embedding_updated: self.embedding_updated()

class AudioCapture(QThread):
    chunk_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, device_name=None, use_mic=False):
        super().__init__()
        self._running, self._paused = True, False
        self.use_mic, self.device_name = use_mic, device_name
        self.device = None
        self._setup_device()
    
    def _setup_device(self):
        try:
            if self.device_name:
                self.device = sc.get_microphone(id=self.device_name, include_loopback=not self.use_mic)
            else:
                self.device = sc.default_microphone() if self.use_mic else sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
        except Exception as e:
            print(f"Error setting up audio device: {e}")
            self.device = sc.default_microphone() if self.use_mic else sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
    
    def change_device(self, device_name, use_mic):
        self.device_name, self.use_mic = device_name, use_mic
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
                        if audio_data.size == 0: continue
                        if len(audio_data.shape) > 1: audio_data = audio_data[:, 0]
                        audio_data = audio_data.flatten().astype(np.float32)
                        max_val = np.max(np.abs(audio_data))
                        if max_val > 1.0: audio_data = audio_data / max_val
                        self.chunk_ready.emit(audio_data)
                    except Exception as e:
                        print(f"Audio recording error: {e}")
                        time.sleep(0.1)
        except Exception as e: print(f"Error initializing audio recorder: {e}")
    
    def pause(self): self._paused = True
    def resume(self): self._paused = False
    def stop(self): self._running = False

class AudioProcessor(QThread):
    tl_updated = pyqtSignal(list)
    
    def __init__(self, encoder, vad_processor, spk_handler, tl_manager, stt_worker=None):
        super().__init__()
        self._running, self._paused = True, False
        self.encoder, self.vad_processor, self.spk_handler, self.tl_manager, self.stt_worker = encoder, vad_processor, spk_handler, tl_manager, stt_worker
        self.spk_handler.set_timeline_manager(self.tl_manager)
        self.buffer = np.zeros(WIN_SAMPLES, dtype=np.float32)
        self.buf_idx = self.buf_full = 0
        self.last_proc_time = self.last_ui_update = 0
        self.pending_vad_tasks, self.pending_segs = {}, {}
        self.pending_ui_update = False
    
    def add_chunk(self, audio_data):
        if self._paused: return
        if self.stt_worker:
            pcm_data = (audio_data * 32767).astype(np.int16).tobytes()
            self.stt_worker.process_audio(pcm_data)
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
        if not self.buf_full: return self.buffer[:self.buf_idx] if self.buf_idx > 0 else None
        window = np.empty(WIN_SAMPLES, dtype=np.float32)
        window[:WIN_SAMPLES - self.buf_idx] = self.buffer[self.buf_idx:]
        window[WIN_SAMPLES - self.buf_idx:] = self.buffer[:self.buf_idx]
        return window
    
    def _proc_window(self):
        window = self._get_window()
        if window is None or len(window) < SAMPLE_RATE * 0.5: return
        timeline_time = self.tl_manager.get_timeline_time()
        seg_start = timeline_time - WINDOW_SIZE
        vad_task_id = self.vad_processor.detect_async(window)
        if vad_task_id:
            seg = Segment(seg_start, WINDOW_SIZE, is_speech=False)
            self.pending_vad_tasks[vad_task_id] = (seg, seg_start, window.copy())
    
    def _proc_vad_results(self):
        while True:
            vad_result = self.vad_processor.get_result()
            if vad_result is None: break
            task_id, is_speech = vad_result
            if task_id in self.pending_vad_tasks:
                seg, seg_time, window = self.pending_vad_tasks.pop(task_id)
                seg.is_speech = is_speech
                if is_speech:
                    emb_task_id = self.encoder.embed_async(window)
                    if emb_task_id: self.pending_segs[emb_task_id] = (seg, seg_time)
                else:
                    self.tl_manager.add_seg(seg)
                    self.pending_ui_update = True
    
    def _proc_emb_res(self):
        while True:
            res = self.encoder.get_res()
            if res is None: break
            task_id, emb = res
            if task_id in self.pending_segs:
                seg, seg_time = self.pending_segs.pop(task_id)
                if emb is not None:
                    spk_id, sim = self.spk_handler.classify_spk(emb, seg_time)
                    seg.spk_id, seg.emb = spk_id, emb
                self.tl_manager.add_seg(seg)
                self.pending_ui_update = True
    
    def pause(self): self._paused = True
    def resume(self): self._paused = False
    
    def reset_tl(self):
        self.spk_handler.reset()
        self.tl_manager.reset()
        self.buffer.fill(0)
        self.buf_idx = self.buf_full = 0
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
    
    def stop(self): self._running = False

class Segment:
    def __init__(self, start_time, duration, spk_id=None, emb=None, is_speech=False):
        self.start_time, self.duration, self.spk_id, self.emb, self.is_speech = start_time, duration, spk_id, emb, is_speech
    @property
    def end_time(self): return self.start_time + self.duration

class Timeline:
    def __init__(self):
        self.segs = []
        self.start_time = self.paused_time = None
        self.total_paused_duration = 0
    
    def start_timeline(self):
        if self.start_time is None:
            self.start_time = time.time()
            self.paused_time = None
            self.total_paused_duration = 0
    
    def pause_timeline(self):
        if self.start_time is not None and self.paused_time is None: self.paused_time = time.time()
    
    def resume_timeline(self):
        if self.paused_time is not None:
            self.total_paused_duration += time.time() - self.paused_time
            self.paused_time = None
    
    def get_timeline_time(self):
        if self.start_time is None: return 0
        current_time = time.time()
        if self.paused_time is not None: return self.paused_time - self.start_time - self.total_paused_duration
        return current_time - self.start_time - self.total_paused_duration
    
    def add_seg(self, seg):
        if self.start_time is not None and self.paused_time is None: self.segs.append(seg)
    
    def update_pending_segments_to_speaker(self, start_time, end_time, new_speaker_id):
        for seg in self.segs:
            if seg.spk_id == "pending" and start_time <= seg.start_time <= end_time:
                seg.spk_id = new_speaker_id
    
    def get_speaker_at_time(self, timestamp):
        for seg in self.segs:
            if seg.is_speech and seg.start_time <= timestamp <= seg.end_time: return seg.spk_id
        return None
    
    def get_dominant_speaker_in_range(self, start_time, end_time):
        speaker_durations = {}
        for seg in self.segs:
            if seg.is_speech and seg.spk_id is not None and seg.spk_id != "pending":
                overlap_start = max(seg.start_time, start_time)
                overlap_end = min(seg.end_time, end_time)
                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    if seg.spk_id not in speaker_durations: speaker_durations[seg.spk_id] = 0
                    speaker_durations[seg.spk_id] += overlap_duration
        if speaker_durations: return max(speaker_durations, key=speaker_durations.get)
        return None
    
    def get_segs(self): return self.segs
    def get_timeline_duration(self): return self.get_timeline_time()
    def reset(self):
        self.segs = []
        self.start_time = self.paused_time = None
        self.total_paused_duration = 0
    
    def reclassify_segs(self, spk_handler):
        for seg in self.segs:
            if seg.is_speech and seg.emb is not None:
                best_spk, best_sim = None, -1.0
                for i, mean_emb in enumerate(spk_handler.mean_embs):
                    if mean_emb is not None:
                        sim = 1.0 - cosine(seg.emb, mean_emb)
                        if sim > best_sim: best_sim, best_spk = sim, i
                seg.spk_id = best_spk

class TranscriptionWindow(QMainWindow):
    def __init__(self, timeline_manager, spk_handler):
        super().__init__()
        self.timeline_manager = timeline_manager
        self.spk_handler = spk_handler
        self.setWindowTitle("TranscriptionWindow")
        self.setGeometry(100, 100, 800, 600)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Arial", 15))
        self.text_edit.setPlaceholderText("waiting...")
        self.text_edit.setAcceptRichText(True)
        layout.addWidget(self.text_edit)
        self.current_recognizing_text = ""
        self.current_speaker = None
        self.setStyleSheet("QMainWindow, QWidget {background-color: #2D2D30; color: #CCCCCC;} QTextEdit {background-color: #1E1E1E; color: #CCCCCC; border: 1px solid #555555;}")
        self.recognized_texts = []  # Store all recognized texts with timestamps
    
    def set_current_speaker(self, speaker_id): self.current_speaker = speaker_id
    
    def get_speaker_color(self, speaker_id):
        if speaker_id == "pending": return PENDING_COLOR
        elif speaker_id is not None and isinstance(speaker_id, int): return SPEAKER_COLORS[speaker_id % len(SPEAKER_COLORS)]
        return "#CCCCCC"
    
    def _find_text_differences(self, old_text, new_text):
        common_prefix_len = 0
        min_len = min(len(old_text), len(new_text))
        for i in range(min_len):
            if old_text[i] == new_text[i]: common_prefix_len += 1
            else: break
        unchanged_text = new_text[:common_prefix_len]
        changed_text = new_text[common_prefix_len:]
        removed_count = len(old_text) - common_prefix_len
        return unchanged_text, changed_text, removed_count
    
    def refresh_colors(self):
        """Refresh all text colors based on current speaker assignments"""
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        self.text_edit.clear()
        
        for text_data in self.recognized_texts:
            self._insert_text_with_speaker_colors(cursor, text_data['original'], text_data['word_timestamps'])
            if text_data.get('translated'):
                cursor.insertText("\n")
                format = QTextCharFormat()
                format.setForeground(QColor("#FFFFFF"))
                cursor.setCharFormat(format)
                cursor.insertText(text_data['translated'])
            cursor.insertText("\n\n")
        
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def update_text_display(self, text_type, content, word_timestamps=None):
        if text_type == "recognizing":
            new_text = content.strip()
            cursor = self.text_edit.textCursor()
            if self.current_recognizing_text:
                unchanged_text, changed_text, removed_count = self._find_text_differences(self.current_recognizing_text, new_text)
                if removed_count > 0:
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    for _ in range(removed_count): cursor.deletePreviousChar()
                if changed_text:
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    format = QTextCharFormat()
                    format.setFontItalic(True)
                    speaker_color = self.get_speaker_color(self.current_speaker)
                    format.setForeground(QColor(speaker_color))
                    cursor.setCharFormat(format)
                    cursor.insertText(changed_text)
            else:
                if new_text:
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    format = QTextCharFormat()
                    format.setFontItalic(True)
                    speaker_color = self.get_speaker_color(self.current_speaker)
                    format.setForeground(QColor(speaker_color))
                    cursor.setCharFormat(format)
                    cursor.insertText(new_text)
            self.current_recognizing_text = new_text
            self.text_edit.setTextCursor(cursor)
        elif text_type == "recognized":
            cursor = self.text_edit.textCursor()
            if self.current_recognizing_text:
                cursor.movePosition(QTextCursor.MoveOperation.End)
                for _ in range(len(self.current_recognizing_text)): cursor.deletePreviousChar()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            
            lines = content.split('\n')
            original_text = lines[0] if lines else content
            translated_text = lines[1] if len(lines) > 1 else ""
            
            # Store recognized text for later refresh
            self.recognized_texts.append({
                'original': original_text,
                'translated': translated_text,
                'word_timestamps': word_timestamps
            })
            
            if word_timestamps:
                self._insert_text_with_speaker_colors(cursor, original_text, word_timestamps)
                if translated_text:
                    cursor.insertText("\n")
                    format = QTextCharFormat()
                    format.setForeground(QColor("#FFFFFF"))
                    cursor.setCharFormat(format)
                    cursor.insertText(translated_text)
            else:
                format = QTextCharFormat()
                format.setForeground(QColor("#CCCCCC"))
                cursor.setCharFormat(format)
                cursor.insertText(content)
            cursor.insertText("\n\n")
            self.current_recognizing_text = ""
            scrollbar = self.text_edit.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def _insert_text_with_speaker_colors(self, cursor, text, word_timestamps):
        words = text.split()
        for i, word in enumerate(words):
            word_info = None
            if i < len(word_timestamps): word_info = word_timestamps[i]
            if word_info:
                dominant_speaker = self.timeline_manager.get_dominant_speaker_in_range(word_info['start_time'], word_info['end_time'])
                speaker_color = self.get_speaker_color(dominant_speaker)
            else: speaker_color = self.get_speaker_color(self.current_speaker)
            format = QTextCharFormat()
            format.setForeground(QColor(speaker_color))
            format.setFontWeight(QFont.Weight.Bold)
            cursor.setCharFormat(format)
            cursor.insertText(word)
            if i < len(words) - 1: cursor.insertText(" ")

class PreRecordingWindow(QMainWindow):
    def __init__(self, spk_handler, encoder, vad_processor, device_name=None, use_mic=False):
        super().__init__()
        self.spk_handler = spk_handler
        self.encoder = encoder
        self.vad_processor = vad_processor
        self.device_name = device_name
        self.use_mic = use_mic
        
        self.setWindowTitle("Pre-Recording Speaker Registration")
        self.setGeometry(100, 100, 800, 600)
        
        # Recording state
        self.is_recording = False
        self.recording_speaker_id = None
        self.recorded_audio = []
        self.audio_capture = None
        
        self._setup_ui()
        
    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Title
        title = QLabel("Register Speakers Before Recording")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        # Speakers grid
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.speakers_layout = QGridLayout()
        scroll_widget.setLayout(self.speakers_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)
        
        # Create speaker slots
        self.speaker_widgets = []
        for i in range(MAX_SPEAKERS):
            widget = self._create_speaker_widget(i)
            self.speaker_widgets.append(widget)
            row = i // 2
            col = i % 2
            self.speakers_layout.addWidget(widget, row, col)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        main_layout.addWidget(close_btn)
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow, QWidget {background-color: #2D2D30; color: #CCCCCC;}
            QPushButton {background: #3F3F46; color: #EEEEEE; border: 1px solid #555555; padding: 8px 15px; margin: 5px;}
            QPushButton:hover {background: #505059;}
            QPushButton:disabled {background: #333337; color: #777777;}
            QLabel {padding: 5px; font-size: 14px;}
            QLineEdit {background: #3F3F46; color: #EEEEEE; border: 1px solid #555555; padding: 5px;}
            QGroupBox {font-weight: bold; border: 2px solid #555555; margin: 10px 0px; padding-top: 10px;}
            QGroupBox::title {subcontrol-origin: margin; left: 10px; padding: 0px 5px 0px 5px;}
        """)
    
    def _create_speaker_widget(self, speaker_id):
        group = QGroupBox(f"Speaker {speaker_id + 1}")
        layout = QVBoxLayout()
        group.setLayout(layout)
        
        # Name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        name_input = QLineEdit()
        name_input.setPlaceholderText(f"Speaker {speaker_id + 1}")
        name_layout.addWidget(name_input)
        layout.addLayout(name_layout)
        
        # Status label
        status_label = QLabel("Not registered")
        status_label.setStyleSheet("color: #888888;")
        layout.addWidget(status_label)
        
        # Embedding count label
        emb_count_label = QLabel("Embeddings: 0")
        layout.addWidget(emb_count_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        record_btn = QPushButton("Record")
        record_btn.clicked.connect(lambda: self._toggle_recording(speaker_id))
        btn_layout.addWidget(record_btn)
        
        fix_btn = QPushButton("Fix")
        fix_btn.clicked.connect(lambda: self._toggle_fix(speaker_id))
        fix_btn.setEnabled(False)
        btn_layout.addWidget(fix_btn)
        
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self._remove_speaker(speaker_id))
        remove_btn.setEnabled(False)
        btn_layout.addWidget(remove_btn)
        
        layout.addLayout(btn_layout)
        
        # Store references
        group.name_input = name_input
        group.status_label = status_label
        group.emb_count_label = emb_count_label
        group.record_btn = record_btn
        group.fix_btn = fix_btn
        group.remove_btn = remove_btn
        group.speaker_id = speaker_id
        
        return group
    
    def _toggle_recording(self, speaker_id):
        widget = self.speaker_widgets[speaker_id]
        
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.recording_speaker_id = speaker_id
            self.recorded_audio = []
            
            widget.record_btn.setText("Stop")
            widget.status_label.setText("Recording...")
            widget.status_label.setStyleSheet("color: #FF4444;")
            
            # Disable other buttons during recording
            for w in self.speaker_widgets:
                if w != widget:
                    w.record_btn.setEnabled(False)
                w.fix_btn.setEnabled(False)
                w.remove_btn.setEnabled(False)
            
            # Start audio capture
            self.audio_capture = AudioCapture(self.device_name, self.use_mic)
            self.audio_capture.chunk_ready.connect(self._on_audio_chunk)
            self.audio_capture.start()
            
        elif self.recording_speaker_id == speaker_id:
            # Stop recording
            self.is_recording = False
            widget.record_btn.setText("Processing...")
            widget.record_btn.setEnabled(False)
            
            # Stop audio capture
            if self.audio_capture:
                self.audio_capture.stop()
                self.audio_capture.wait()
                self.audio_capture = None
            
            # Process recorded audio
            QTimer.singleShot(100, lambda: self._process_recording(speaker_id))
    
    def _on_audio_chunk(self, chunk):
        if self.is_recording:
            self.recorded_audio.append(chunk)
    
    def _process_recording(self, speaker_id):
        widget = self.speaker_widgets[speaker_id]
        
        if not self.recorded_audio:
            widget.status_label.setText("No audio recorded")
            widget.status_label.setStyleSheet("color: #FF4444;")
            widget.record_btn.setText("Record")
            widget.record_btn.setEnabled(True)
            self._enable_all_buttons()
            return
        
        # Concatenate audio
        full_audio = np.concatenate(self.recorded_audio)
        
        # Process audio in windows to extract embeddings
        embeddings = []
        window_size = WIN_SAMPLES
        step_size = window_size // 2  # 50% overlap
        
        for i in range(0, len(full_audio) - window_size, step_size):
            window = full_audio[i:i + window_size]
            
            # Check if window contains speech
            task_id = self.vad_processor.detect_async(window)
            if task_id:
                # Wait for VAD result
                timeout = 0
                while timeout < 50:  # 5 seconds timeout
                    result = self.vad_processor.get_result()
                    if result and result[0] == task_id:
                        if result[1]:  # Contains speech
                            # Extract embedding
                            emb_task_id = self.encoder.embed_async(window)
                            if emb_task_id:
                                # Wait for embedding result
                                emb_timeout = 0
                                while emb_timeout < 50:
                                    emb_result = self.encoder.get_res()
                                    if emb_result and emb_result[0] == emb_task_id:
                                        if emb_result[1] is not None:
                                            embeddings.append(emb_result[1])
                                        break
                                    time.sleep(0.1)
                                    emb_timeout += 1
                        break
                    time.sleep(0.1)
                    timeout += 1
        
        if embeddings:
            # Register speaker with embeddings
            name = widget.name_input.text() or f"Speaker {speaker_id + 1}"
            success = self.spk_handler.register_speaker(speaker_id, embeddings, name)
            
            if success:
                widget.status_label.setText(f"Registered ({len(embeddings)} segments)")
                widget.status_label.setStyleSheet("color: #44FF44;")
                widget.emb_count_label.setText(f"Embeddings: {len(embeddings)}")
                widget.fix_btn.setEnabled(True)
                widget.remove_btn.setEnabled(True)
                widget.fix_btn.setText("Unfix" if speaker_id in self.spk_handler.fixed_speakers else "Fix")
            else:
                widget.status_label.setText("Registration failed")
                widget.status_label.setStyleSheet("color: #FF4444;")
        else:
            widget.status_label.setText("No speech detected")
            widget.status_label.setStyleSheet("color: #FFAA44;")
        
        widget.record_btn.setText("Record")
        widget.record_btn.setEnabled(True)
        self._enable_all_buttons()
        self.recorded_audio = []
    
    def _toggle_fix(self, speaker_id):
        widget = self.speaker_widgets[speaker_id]
        
        if speaker_id in self.spk_handler.fixed_speakers:
            self.spk_handler.unfix_speaker(speaker_id)
            widget.fix_btn.setText("Fix")
            widget.status_label.setText("Unfixed (updatable)")
            widget.status_label.setStyleSheet("color: #FFAA44;")
        else:
            self.spk_handler.fix_speaker(speaker_id)
            widget.fix_btn.setText("Unfix")
            widget.status_label.setText("Fixed (locked)")
            widget.status_label.setStyleSheet("color: #44FFFF;")
    
    def _remove_speaker(self, speaker_id):
        widget = self.speaker_widgets[speaker_id]
        
        reply = QMessageBox.question(self, "Confirm Removal", 
                                    f"Remove all embeddings for Speaker {speaker_id + 1}?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            if self.spk_handler.remove_speaker(speaker_id):
                widget.status_label.setText("Not registered")
                widget.status_label.setStyleSheet("color: #888888;")
                widget.emb_count_label.setText("Embeddings: 0")
                widget.fix_btn.setEnabled(False)
                widget.remove_btn.setEnabled(False)
                widget.fix_btn.setText("Fix")
                widget.name_input.clear()
    
    def _enable_all_buttons(self):
        for widget in self.speaker_widgets:
            widget.record_btn.setEnabled(True)
            if widget.speaker_id in self.spk_handler.active_spks:
                widget.fix_btn.setEnabled(True)
                widget.remove_btn.setEnabled(True)
    
    def update_display(self):
        """Update display to reflect current speaker states"""
        for widget in self.speaker_widgets:
            spk_id = widget.speaker_id
            if spk_id in self.spk_handler.active_spks:
                emb_count = len(self.spk_handler.spk_embs[spk_id])
                widget.emb_count_label.setText(f"Embeddings: {emb_count}")
                
                if spk_id in self.spk_handler.fixed_speakers:
                    widget.status_label.setText("Fixed (locked)")
                    widget.status_label.setStyleSheet("color: #44FFFF;")
                    widget.fix_btn.setText("Unfix")
                else:
                    widget.status_label.setText(f"Registered ({emb_count} segments)")
                    widget.status_label.setStyleSheet("color: #44FF44;")
                    widget.fix_btn.setText("Fix")
                
                widget.fix_btn.setEnabled(True)
                widget.remove_btn.setEnabled(True)
                
                # Update name if available
                name = self.spk_handler.speaker_names.get(spk_id)
                if name:
                    widget.name_input.setText(name)
            else:
                widget.status_label.setText("Not registered")
                widget.status_label.setStyleSheet("color: #888888;")
                widget.emb_count_label.setText("Embeddings: 0")
                widget.fix_btn.setEnabled(False)
                widget.remove_btn.setEnabled(False)
                widget.fix_btn.setText("Fix")

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
        if embeddings is None or len(embeddings) < 2: return
        embeddings_2d = self.pca.fit_transform(embeddings)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        unique_labels = set(labels)
        for label in unique_labels:
            mask = np.array(labels) == label
            points = embeddings_2d[mask]
            if label == -1: 
                ax.scatter(points[:, 0], points[:, 1], c=PENDING_COLOR, alpha=0.7, s=50, label=f"Pending ({len(points)})")
            else:
                color = SPEAKER_COLORS[label % len(SPEAKER_COLORS)]
                name = self.spk_handler.get_speaker_name(label)
                fixed_marker = "*" if label in self.spk_handler.fixed_speakers else "o"
                ax.scatter(points[:, 0], points[:, 1], c=color, alpha=0.7, s=50, 
                          marker=fixed_marker, label=f"{name} ({len(points)})")
        mean_embeddings, mean_labels = [], []
        for spk_id in self.spk_handler.active_spks:
            if self.spk_handler.mean_embs[spk_id] is not None:
                mean_embeddings.append(self.spk_handler.mean_embs[spk_id])
                mean_labels.append(spk_id)
        if mean_embeddings:
            mean_embeddings_2d = self.pca.transform(np.array(mean_embeddings))
            for mean_point, spk_id in zip(mean_embeddings_2d, mean_labels):
                color = SPEAKER_COLORS[spk_id % len(SPEAKER_COLORS)]
                ax.scatter(mean_point[0], mean_point[1], c=color, marker='*', s=200, edgecolors='black', linewidth=1)
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_title('Speaker Embeddings Visualization (PCA)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fixed_count = len(self.spk_handler.fixed_speakers)
        stats_text = f"Total: {len(embeddings)}, Active: {len(self.spk_handler.active_spks)}, Fixed: {fixed_count}, Pending: {len(self.spk_handler.pending_embs)}\nPending: {self.spk_handler.pending_enabled}, Update: {self.spk_handler.embedding_update_enabled}\nMin cluster size: {MIN_CLUSTER_SIZE}, Embedding update thresh: {EMBEDDING_UPDATE_THRESHOLD}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        self.canvas.draw()

class TimelineUI(QWidget):
    def __init__(self, spk_handler):
        super().__init__()
        self.spk_handler = spk_handler
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
        if not self.segs: new_width = 800
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
        if not self.segs: return
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
            if seg.end_time < visible_start or seg.start_time > visible_end: continue
            x_start = int(seg.start_time * self.pixels_per_second)
            x_end = int(seg.end_time * self.pixels_per_second)
            if x_start > width or x_end < 0: continue
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
            name = self.spk_handler.get_speaker_name(i)
            if i in self.spk_handler.fixed_speakers:
                name += " [Fixed]"
            painter.drawText(10, y, name)
        y = int(self.max_spks * layer_h + layer_h // 2 + 5)
        painter.drawText(10, y, "Pending")
        y = int((self.max_spks + 1) * layer_h + layer_h // 2 + 5)
        painter.drawText(10, y, "Non-Speech")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("speech-to-text diarization")
        self.encoder = self.vad_processor = self.audio_capture = self.audio_proc = self.stt_worker = None
        self.spk_handler = SpeakerHandler()
        self.tl_manager = Timeline()
        self.is_recording = False
        self.visualization_window = self.transcription_window = self.pre_recording_window = None
        self.models_loaded = {"encoder": False, "vad": False}
        self.spk_handler.set_embedding_callback(self._on_embeddings_updated)
        self.spk_handler.set_speaker_changed_callback(self._on_speaker_changed)
        self._setup_ui()
        QTimer.singleShot(500, self._init_app)
    
    def _on_speaker_changed(self, speaker_id):
        if self.stt_worker: self.stt_worker.set_current_speaker(speaker_id)
        if self.transcription_window: self.transcription_window.set_current_speaker(speaker_id)
    
    def _get_audio_devices(self):
        try:
            microphones = [(mic.name, True) for mic in sc.all_microphones()]
            speakers = [(spk.name, False) for spk in sc.all_speakers()]
            return microphones + speakers
        except Exception as e:
            print(f"Error getting audio devices: {e}")
            return []
    
    def _refresh_devices(self):
        self.device_combo.clear()
        devices = self._get_audio_devices()
        for device_name, is_mic in devices:
            device_type = "Microphone" if is_mic else "Speaker"
            display_name = f"[{device_type}] {device_name}"
            self.device_combo.addItem(display_name, (device_name, is_mic))
        if self.device_combo.count() > 0:
            try:
                default_speaker_name = str(sc.default_speaker().name)
                for i in range(self.device_combo.count()):
                    device_name, is_mic = self.device_combo.itemData(i)
                    if device_name == default_speaker_name and not is_mic:
                        self.device_combo.setCurrentIndex(i)
                        break
                else: self.device_combo.setCurrentIndex(0)
            except: self.device_combo.setCurrentIndex(0)
    
    def _apply_device_change(self):
        if self.device_combo.currentIndex() < 0: return
        device_name, is_mic = self.device_combo.currentData()
        was_recording = self.is_recording
        if was_recording: self._toggle_recording()
        if self.audio_capture:
            self.audio_capture.stop()
            self.audio_capture.wait()
        self.audio_capture = AudioCapture(device_name=device_name, use_mic=is_mic)
        self.audio_capture.pause()
        self.audio_capture.chunk_ready.connect(self.audio_proc.add_chunk)
        self.audio_capture.start()
        device_type = "Microphone" if is_mic else "Speaker"
        self.status_label.setText(f"Audio device changed to: [{device_type}] {device_name}")
        if was_recording: QTimer.singleShot(100, self._toggle_recording)
    
    def _on_embeddings_updated(self):
        if self.visualization_window and self.visualization_window.isVisible(): self.visualization_window.update_plot()
    
    def _setup_ui(self):
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)
        self.status_label = QLabel("Preparing...")
        layout.addWidget(self.status_label)
        self.scroll_area = QScrollArea()
        self.timeline = TimelineUI(self.spk_handler)
        self.scroll_area.setWidget(self.timeline)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(self.scroll_area, 1)
        
        btn_layout = QHBoxLayout()
        self.start_pause_btn = QPushButton("Start")
        self.start_pause_btn.clicked.connect(self._toggle_recording)
        self.start_pause_btn.setEnabled(False)
        btn_layout.addWidget(self.start_pause_btn)
        
        self.pre_recording_btn = QPushButton("Pre-Recording")
        self.pre_recording_btn.clicked.connect(self._show_pre_recording)
        self.pre_recording_btn.setEnabled(False)
        btn_layout.addWidget(self.pre_recording_btn)
        
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
        self.transcription_btn = QPushButton("Show Transcription")
        self.transcription_btn.clicked.connect(self._show_transcription)
        self.transcription_btn.setEnabled(False)
        btn_layout.addWidget(self.transcription_btn)
        self.pending_btn = QPushButton("Disable Pending")
        self.pending_btn.clicked.connect(self._toggle_pending)
        self.pending_btn.setEnabled(False)
        btn_layout.addWidget(self.pending_btn)
        self.embedding_update_btn = QPushButton("Disable Embedding Update")
        self.embedding_update_btn.clicked.connect(self._toggle_embedding_update)
        self.embedding_update_btn.setEnabled(False)
        btn_layout.addWidget(self.embedding_update_btn)
        layout.addLayout(btn_layout)
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
        self._refresh_devices()
        self.scroll_timer = QTimer()
        self.scroll_timer.timeout.connect(self._auto_scroll)
        self.scroll_timer.start(2000)
        self.setStyleSheet("""QMainWindow, QWidget {background-color: #2D2D30; color: #CCCCCC;}
            QPushButton {background: #3F3F46; color: #EEEEEE; border: 1px solid #555555; padding: 8px 15px; margin: 5px;}
            QPushButton:hover {background: #505059;} QPushButton:disabled {background: #333337; color: #777777;}
            QLabel {padding: 5px; font-size: 14px;} QScrollArea {border: 1px solid #555555;}
            QGroupBox {font-weight: bold; border: 2px solid #555555; margin: 10px 0px; padding-top: 10px;}
            QGroupBox::title {subcontrol-origin: margin; left: 10px; padding: 0px 5px 0px 5px;}
            QComboBox {background: #3F3F46; color: #EEEEEE; border: 1px solid #555555; padding: 5px 10px; margin: 5px;}
            QComboBox::drop-down {border: none;} QComboBox::down-arrow {image: none; border: none;}""")
    
    def _auto_scroll(self):
        if self.is_recording:
            scrollbar = self.scroll_area.horizontalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def _toggle_recording(self):
        if not self.is_recording:
            self.tl_manager.start_timeline()
            self.tl_manager.resume_timeline()
            if self.audio_capture: self.audio_capture.resume()
            if self.audio_proc: self.audio_proc.resume()
            if self.stt_worker: self.stt_worker.start_stt_with_timeline_sync()
            self.is_recording = True
            self.start_pause_btn.setText("Pause")
            self.status_label.setText("Recording... (Timeline + STT)")
        else:
            self.tl_manager.pause_timeline()
            if self.audio_capture: self.audio_capture.pause()
            if self.audio_proc: self.audio_proc.pause()
            self.is_recording = False
            self.start_pause_btn.setText("Resume")
            self.status_label.setText("Paused")
    
    def _reset_tl(self):
        if self.audio_proc: self.audio_proc.reset_tl()
        if self.stt_worker:
            self.stt_worker.stt_start_time = self.stt_worker.timeline_start_time = None
            self.stt_worker.time_offset = 0.0
        self.is_recording = False
        self.start_pause_btn.setText("Start")
        self.status_label.setText("Timeline has been reset.")
    
    def _recluster(self):
        if not self.audio_proc: return
        total_embeddings = self.spk_handler.get_total_embedding_count()
        if total_embeddings < 50:
            QMessageBox.information(self, "Information", f"Not enough embeddings for reclustering.\nCurrent embeddings: {total_embeddings}\nRequired embeddings: 50 or more")
            return
        active_speakers = len(self.spk_handler.active_spks)
        fixed_speakers = len(self.spk_handler.fixed_speakers)
        cluster_count, ok = QInputDialog.getInt(self, "Recluster Speakers", 
            f"Enter number of speakers:\nTotal embeddings: {total_embeddings}\nCurrently detected speakers: {active_speakers}\nFixed speakers: {fixed_speakers}",
            value=max(2, active_speakers), min=max(2, fixed_speakers + 1), max=MAX_SPEAKERS)
        if ok:
            success = self.audio_proc.recluster_spks(cluster_count)
            if success: 
                self.status_label.setText(f"Reclustered {total_embeddings} embeddings into {cluster_count} speakers.")
                # Refresh transcription window colors after reclustering
                if self.transcription_window and self.transcription_window.isVisible():
                    self.transcription_window.refresh_colors()
            else: QMessageBox.warning(self, "Error", "Reclustering failed.")
    
    def _toggle_pending(self):
        if not self.spk_handler: return
        is_enabled = self.spk_handler.toggle_pending()
        if is_enabled:
            self.pending_btn.setText("Disable Pending")
            self.status_label.setText("Pending feature enabled.")
        else:
            self.pending_btn.setText("Enable Pending")
            self.status_label.setText("Pending feature disabled.")
    
    def _toggle_embedding_update(self):
        if not self.spk_handler: return
        is_enabled = self.spk_handler.toggle_embedding_update()
        if is_enabled:
            self.embedding_update_btn.setText("Disable Embedding Update")
            self.status_label.setText("Embedding update enabled.")
        else:
            self.embedding_update_btn.setText("Enable Embedding Update")
            self.status_label.setText("Embedding update disabled.")
    
    def _show_visualization(self):
        if self.visualization_window is None: self.visualization_window = EmbeddingVisualizationWindow(self.spk_handler)
        self.visualization_window.show()
        self.visualization_window.raise_()
        self.visualization_window.activateWindow()
    
    def _show_transcription(self):
        if self.transcription_window is None:
            self.transcription_window = TranscriptionWindow(self.tl_manager, self.spk_handler)
            if self.stt_worker: self.stt_worker.update_text.connect(self.transcription_window.update_text_display)
        self.transcription_window.show()
        self.transcription_window.raise_()
        self.transcription_window.activateWindow()
    
    def _show_pre_recording(self):
        device_name, use_mic = None, False
        if self.device_combo.currentIndex() >= 0:
            device_name, use_mic = self.device_combo.currentData()
        
        if self.pre_recording_window is None:
            self.pre_recording_window = PreRecordingWindow(
                self.spk_handler, self.encoder, self.vad_processor, 
                device_name, use_mic
            )
        else:
            # Update device settings
            self.pre_recording_window.device_name = device_name
            self.pre_recording_window.use_mic = use_mic
        
        self.pre_recording_window.update_display()
        self.pre_recording_window.show()
        self.pre_recording_window.raise_()
        self.pre_recording_window.activateWindow()
    
    def _init_app(self):
        self.resize(1400, 900)
        device = "cuda" if DEVICE_PREF == "cuda" and torch.cuda.is_available() else "cpu"
        if DEVICE_PREF == "cuda" and not torch.cuda.is_available(): print("CUDA not available, falling back to CPU")
        else: print(f"Using {device.upper()} device")
        self.encoder = SpeechBrainEncoder(device)
        self.encoder.model_loaded.connect(self._on_encoder_loaded)
        self.encoder.start()
        self.vad_processor = SileroVAD(device, VAD_THRESH)
        self.vad_processor.model_loaded.connect(self._on_vad_loaded)
        self.vad_processor.start()
        self.stt_worker = STTWorker(self.tl_manager)
        self.stt_worker.start()
        self.status_label.setText("Loading models... (Encoder + VAD + STT)")
    
    def _on_encoder_loaded(self):
        self.models_loaded["encoder"] = True
        self._check_models_ready()
    
    def _on_vad_loaded(self):
        self.models_loaded["vad"] = True
        self._check_models_ready()
    
    def _check_models_ready(self):
        if all(self.models_loaded.values()):
            device_name, use_mic = None, False
            if self.device_combo.currentIndex() >= 0: device_name, use_mic = self.device_combo.currentData()
            self.audio_capture = AudioCapture(device_name=device_name, use_mic=use_mic)
            self.audio_proc = AudioProcessor(self.encoder, self.vad_processor, self.spk_handler, self.tl_manager, self.stt_worker)
            self.audio_capture.pause()
            self.audio_proc.pause()
            self.audio_capture.chunk_ready.connect(self.audio_proc.add_chunk)
            self.audio_proc.tl_updated.connect(self.timeline.update_segs)
            self.audio_proc.start()
            self.audio_capture.start()
            self.start_pause_btn.setEnabled(True)
            self.pre_recording_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)
            self.recluster_btn.setEnabled(True)
            self.viz_btn.setEnabled(True)
            self.transcription_btn.setEnabled(True)
            self.pending_btn.setEnabled(True)
            self.embedding_update_btn.setEnabled(True)
            self.apply_device_btn.setEnabled(True)
            if device_name:
                device_type = "Microphone" if use_mic else "Speaker"
                self.status_label.setText(f"Ready - Current device: [{device_type}] {device_name}")
            else: self.status_label.setText("Ready - Press Start button to begin")
    
    def closeEvent(self, event):
        if self.visualization_window: self.visualization_window.close()
        if self.transcription_window: self.transcription_window.close()
        if self.pre_recording_window: self.pre_recording_window.close()
        if self.audio_capture: self.audio_capture.stop()
        if self.audio_proc: self.audio_proc.stop()
        if self.encoder: self.encoder.stop_proc()
        if self.vad_processor: self.vad_processor.stop_processing()
        if self.stt_worker: self.stt_worker.stop()
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__": main()
