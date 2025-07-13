This is draft. I'll explain how I implemented real-time speaker diarization in detail later.

---

# Real-time Speaker Diarization Timeline

This application provides a real-time speaker diarization timeline using advanced deep learning models. It captures system audio (or microphone input), detects voice activity, generates speaker embeddings, and visualizes who is speaking and when on a dynamic timeline. It features an intelligent system for automatically identifying and adding new speakers.

<img width="800" height="630" alt="image" src="https://github.com/user-attachments/assets/d1df87f9-83ed-4fd1-a29a-ac6c11114d09" />

## Features

- **Real-time Visualization**: A continuously updating timeline that maps speech segments to different speakers.
- **High-Quality Speaker Embeddings**: Utilizes the **ECAPA-TDNN** model (via SpeechBrain) to generate robust speaker embeddings.
- **Accurate Voice Activity Detection (VAD)**: Employs the **Silero VAD** model for efficient and reliable speech detection.
- **Automatic Speaker Promotion**: A novel "pending" system that automatically identifies and promotes new speakers when they speak consistently.
- **Dynamic Re-clustering**: Manually re-cluster all collected speech embeddings to refine speaker assignments at any time.
- **Embedding Visualization**: An interactive PCA plot to visualize the distribution of speaker embeddings in 2D space, helping to understand the model's classifications.
- **Configurable**: Easily tweak parameters like similarity thresholds, buffer sizes, and model settings directly in the source code.
- **Audio Source Selection**: Can capture either system audio (loopback) or microphone input.

## Core Logic Explained

The application's intelligence lies in how it assigns speech segments to speakers and how it discovers new ones. This is a two-part process:

### 1. Cosine Similarity-Based Speaker Assignment

When a segment of audio is identified as speech by the VAD, an embedding vector is generated using the ECAPA-TDNN model. This vector is a numerical representation of the speaker's voice characteristics. To determine which speaker this segment belongs to, the system performs the following steps:

1.  **Calculate Similarity**: The new embedding is compared against the pre-computed "mean embedding" for every known speaker. The comparison metric is **Cosine Similarity**, which measures the cosine of the angle between two vectors. A similarity score of `1.0` means the voices are identical in characteristics, while `0.0` means they are completely different.

2.  **Decision Thresholds**: The system uses a multi-level thresholding logic to make a decision:
    *   **High Similarity (`>= EMBEDDING_UPDATE_THRESHOLD`, e.g., 0.4)**: If the similarity to the best-matching speaker is very high, the segment is confidently assigned to that speaker. The new embedding is then added to that speaker's collection, and their "mean embedding" is recalculated (using the median for robustness against outliers). This allows the speaker's profile to adapt over time.
    *   **Medium Similarity (`>= PENDING_THRESHOLD`, e.g., 0.2)**: If the similarity is moderate, the segment is still assigned to the best-matching speaker. However, the speaker's mean embedding is **not** updated. This prevents a potentially incorrect assignment from "polluting" a well-defined speaker profile.
    *   **Low Similarity (`< PENDING_THRESHOLD`)**: If the new embedding does not closely match any known speaker, it is considered to be from an unknown or new speaker. The segment is assigned to a special **"Pending"** pool for further analysis.

### 2. Automatic Speaker Promotion Logic

The "Pending" system is designed to automatically discover and add new speakers without manual intervention.

1.  **Collect Pending Embeddings**: All embeddings that fail to match an existing speaker are collected in a temporary buffer.

2.  **Trigger Clustering**: Once the number of pending embeddings reaches a minimum size (`MIN_CLUSTER_SIZE`, e.g., 30 samples), an automatic analysis is triggered. This ensures we have enough data to make a reliable decision.

3.  **Find Cohesive Groups**: The system uses **Agglomerative Clustering** with a `cosine` distance metric on the pending embeddings. The goal is to find a dense, cohesive group of embeddings that likely belong to a single new speaker.

4.  **Promote New Speaker**: If the clustering algorithm finds a cluster that is large enough (`>= MIN_CLUSTER_SIZE`) and internally consistent (within `AUTO_CLUSTER_DISTANCE_THRESHOLD`), that cluster is "promoted" to a new speaker:
    *   A new speaker ID is created.
    *   All embeddings from the promoted cluster are assigned to this new speaker.
    *   A mean embedding is calculated for this new speaker.
    *   The timeline is updated, re-assigning all corresponding segments from "Pending" to the new speaker ID.
    *   The pending buffer is cleared, ready to discover the next new speaker.

This promotion logic allows the application to start with no known speakers and dynamically add them as they are detected in the audio stream.

## Requirements

- Python 3.8+
- PyTorch (`torch`, `torchaudio`)
- PyQt6
- soundcard
- numpy
- scikit-learn
- scipy
- matplotlib
- speechbrain

## Installation
**Install the required packages:**
    ```bash
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install soundcard numpy PyQt6 scikit-learn scipy matplotlib speechbrain
    ```
    *(Note: Adjust the PyTorch installation command based on your system's CUDA version or if you're using CPU only. See the [PyTorch website](https://pytorch.org/get-started/locally/) for details.)*

## How to Run

Simply execute the main Python script:

```bash
python rt_timeline.py
```

The first time you run the application, it will download the necessary VAD and speaker embedding models, which may take a few minutes. These are cached for future runs.

## Configuration

Key parameters can be adjusted at the top of the `rt_timeline.py` file:

- `DEVICE_PREF`: Set to `"cuda"` to use a GPU or `"cpu"`.
- `VAD_THRESH`: The confidence threshold for the Silero VAD model (0.0 to 1.0).
- `PENDING_THRESHOLD`: The cosine similarity score below which an embedding is sent to the "Pending" pool.
- `EMBEDDING_UPDATE_THRESHOLD`: The cosine similarity score above which a speaker's mean embedding is updated.
- `MIN_CLUSTER_SIZE`: The minimum number of pending embeddings required to attempt a new speaker promotion.
- `AUTO_CLUSTER_DISTANCE_THRESHOLD`: The maximum cosine distance for clustering pending embeddings.
