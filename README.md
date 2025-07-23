
# utterr — Real-time Speaker Diarization Timeline

> 📌 *All code was written with the help of Claude.*
> 
> *Unfortunately, as a programming beginner, I could never have done it alone 😢*




An example project that extracts speaker embeddings using SpeechBrain's ECAPA-TDNN model, performs real-time speaker identification and separation, and visualizes the results on a timeline.

This project serves as a demo and example code showcasing how to implement a real-time speaker diarization system using `SpeechBrain`'s models and toolkit.

[![Video](https://img.youtube.com/vi/0DTFxcHnJ14/maxresdefault.jpg)](https://youtu.be/0DTFxcHnJ14?si=1QqzzcVz6zpmH_vy)

**▶️ Demo Video(Click Above for the Demo Video)**  


## Key Features

*   **Real-time Audio Capture**: Uses the `SoundCard` library to capture microphone or system audio (loopback) in real-time. SoundCard can capture speaker output via loopback without additional setup, making it more useful than PyAudio in my opinion.
*   **Voice Activity Detection (VAD)**: Uses Silero VAD to accurately detect speech segments in audio.
*   **Speaker Embedding Extraction**: Extracts speaker embeddings (voice feature vectors) from each speech segment using SpeechBrain's pre-trained ECAPA-TDNN model.
*   **Dynamic Speaker Clustering**: When a new speaker appears, they are first classified as 'Pending', then automatically clustered and registered as a new speaker once sufficient data is accumulated.
*   **Manual Reclustering**: If clustering goes wrong, you can manually reclassify all accumulated embeddings to reassign speakers.
*   **Real-time Timeline Visualization**: Uses PyQt6 to display in real-time which speaker spoke when on a timeline.
*   **Embedding Distribution Visualization**: Visualizes high-dimensional embeddings in 2D using PCA to show how speaker clusters are distributed.
<img width="800" height="630" alt="image" src="https://github.com/user-attachments/assets/d1df87f9-83ed-4fd1-a29a-ac6c11114d09" />
## 🛠 Installation and Usage

### Requirements

*   **Windows**: SoundCard's loopback functionality only works on Windows!
*   **Python**: 3.10 or higher
*   **Conda**

### Installation Process

1.  **Create and activate Conda environment**
    ```bash
    conda create -n rtt_env python=3.10
    conda activate rtt_env
    ```

2.  **Install required packages**

    It's recommended to install PyTorch according to your system environment (CUDA support).
    *   **CPU version:**
        ```bash
        pip install torch torchaudio soundcard numpy PyQt6 scikit-learn matplotlib speechbrain
        ```
    *   **GPU (CUDA) version:** (if you have NVIDIA GPU)
        Check the installation command for your CUDA version on the [PyTorch official website](https://pytorch.org/get-started/locally/).

### Execution

Run the program with the following command **(Run as administrator on first launch)**:
```bash
python rt_timeline.py
```

When the program starts, it will download the models. Once the "Ready - Press Start button to begin" message appears, click the "Start" button to begin real-time speaker diarization.

## ⚙️ How It Works

The system processes audio streams using a sliding window approach, dividing them into short windows and sequentially processing them to identify speakers.

### 1. Audio Window Processing
Creates and processes 1-second (`WINDOW_SIZE`) audio windows at 0.1-second (`WINDOW_PROCESS_INTERVAL`) intervals.

### 2. Voice Activity Detection (VAD)
Each 1-second audio window is passed through the **Silero VAD** model to determine whether it contains speech or non-speech (is_speech).

### 3. Embedding Extraction
Windows identified as speech are passed through `SpeechBrain`'s **ECAPA-TDNN model** to extract 'embeddings' - high-dimensional vectors representing the speaker's voice characteristics.

### 4. 'Pending' Processing
When a new embedding has low similarity with all existing registered speakers, it's temporarily assigned to the 'Pending' queue. This is a process of collecting unidentified new speakers for automatic clustering promotion. 

Specifically, when calculating cosine similarity between a new embedding and all existing speaker representatives, if even the highest similarity is below `PENDING_THRESHOLD` (referred to as `self.change_thresh` in code, default 0.3), it's processed as 'Pending'.

### 5. New Speaker Promotion (Automatic Clustering)
When the 'Pending' queue accumulates more than a certain number of embeddings (`MIN_CLUSTER_SIZE`), **scikit-learn**'s `Agglomerative Clustering` is performed to find new speaker clusters. 

This clustering groups embeddings based on cosine distance and automatically determines the number of clusters through `distance_threshold` (usually dozens of clusters may be created). Lower values create clusters with lower variance, forming tightly packed regions suitable as reference points for individual speakers with minimal outliers. If the largest cluster meets the `MIN_CLUSTER_SIZE` criteria, that cluster is 'promoted' to a new speaker and registered on the timeline.

```python
# _find_cohesive_group method in rt_timeline.py
def _find_cohesive_group(self):
    # ...
    try:
        # AgglomerativeClustering performs hierarchical clustering.
        # Here, n_clusters=None and distance_threshold is specified to automatically determine
        # the number of clusters based on distance according to data structure.
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=AUTO_CLUSTER_DISTANCE_THRESHOLD,  # 0.6
            metric='cosine',
            linkage='average' # Uses average distance between clusters
        )
        labels = clustering.fit_predict(np.array(self.pending_embs))

        # The largest cluster among generated clusters is considered as a new speaker candidate.
        # ...
```

*(Speaker utterances generally form spherical clusters, and I believe the AgglomerativeClustering method is well-suited for speaker utterance spherical clusters.)*

### 6. Speaker Classification (Cosine Similarity-based)
When a new embedding is not processed as `PENDING_THRESHOLD`, it determines which existing speaker it belongs to. This is done by calculating **cosine similarity** between each speaker cluster's representative embedding (median) and the new embedding. The embedding is assigned to the speaker with the highest similarity. Once assigned, the representative embedding value of that cluster is updated (only when above EMBEDDING_UPDATE_THRESHOLD).

```python
def classify_spk(self, emb, seg_time):
    # ...
    # Normalize new embedding vector
    emb_norm = emb / np.linalg.norm(emb)

    # Get and normalize existing speakers' representative embedding matrix
    mean_embs_matrix = np.array(active_mean_embs)
    mean_embs_norm = mean_embs_matrix / np.linalg.norm(mean_embs_matrix, axis=1, keepdims=True)

    similarities = np.dot(mean_embs_norm, emb_norm)

    # Find the speaker with highest similarity
    best_idx = np.argmax(similarities)
    best_sim = similarities[best_idx]
    best_spk = active_spk_ids[best_idx]

    # If similarity exceeds threshold, classify as that speaker
    if best_sim >= self.change_thresh: # PENDING_THRESHOLD
        spk_id = best_spk
        self.curr_spk = spk_id
        # If similarity exceeds EMBEDDING_UPDATE_THRESHOLD, update representative embedding too
        # ...
        return spk_id, best_sim
    else:
        # If similarity is low, process as 'Pending' (step 4)
        # ...
        return "pending", best_sim
```

### 7. Manual Reclustering
Through the "Recluster Speakers" button, users can re-perform complete clustering on all collected embeddings (existing speakers + Pending) according to a user-specified number of speakers (`n_clusters`). This allows users to directly resolve errors from the automatic clustering process, such as misclassification or one speaker being split into multiple speakers.

## ⚠️ Issues and Troubles

### Parameter Dependency
Key parameters like `PENDING_THRESHOLD` and `AUTO_CLUSTER_DISTANCE_THRESHOLD` may require adjustment for optimal values depending on the usage environment (field conditions, noise levels, number of speakers).

### Similar Voice Distinction
Cosine similarity-based classification may not effectively distinguish speakers with very similar vocal timbres when they speak simultaneously.

### Threshold Setting Dilemma
*   When many speakers with similar voices are expected, `PENDING_THRESHOLD` and `AUTO_CLUSTER_DISTANCE_THRESHOLD` should be set low.
*   However, this can cause a single speaker to be incorrectly promoted as multiple new speakers when their speech characteristics change slightly. The **reclustering** feature was added to correct such issues post-hoc.

### Robustness
Due to the above reasons, this is a demo version that is not perfectly 'robust' against various exceptional situations.
