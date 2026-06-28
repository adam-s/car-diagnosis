# Architecture Diagrams

Mermaid diagrams for the whole `car-diagnosis` system: the end-to-end pipeline, the
per-platform scrapers, the clean+segment cascade, training, and the abstracted
inference core shared by the CLI and the local web app.

---

## 1. End-to-end overview

```mermaid
flowchart LR
    subgraph SCRAPE
        YT[YouTube<br/>yt-dlp]
        TT[TikTok<br/>Camoufox + yt-dlp]
        RD[Reddit<br/>old.reddit + yt-dlp]
    end

    subgraph CLEAN[Clean + Segment]
        CAS[Cascade:<br/>energy - VAD - flatness - music]
        SEG[Mechanical segments<br/>clip WAVs]
    end

    subgraph LABEL[Label]
        OCR[OCR consensus<br/>Apple Vision]
        TXT[Native text<br/>chapters / captions / comments]
        FUSE[LLM fusion + tiering<br/>gold / silver / bronze]
    end

    subgraph TRAIN[Train]
        EMB[CLAP embeddings]
        HEADS[Linear heads<br/>kind / knock / cause]
        CAL[Calibration<br/>confidence bands]
    end

    subgraph CLASSIFY[Classify]
        CORE[predict core]
        CLI[CLI]
        WEB[Local web upload]
    end

    YT --> CAS
    TT --> CAS
    RD --> CAS
    CAS --> SEG
    SEG --> OCR
    SEG --> TXT
    OCR --> FUSE
    TXT --> FUSE
    FUSE --> EMB
    EMB --> HEADS
    HEADS --> CAL
    CAL -->|trained model .joblib| CORE
    CLI --> CORE
    WEB --> CORE
```

---

## 2. Per-platform scrape paths

```mermaid
flowchart TD
    subgraph YouTube
        Y1[discover.py<br/>yt-dlp ytsearch] --> Y2[batch.py<br/>download audio]
        Y2 --> Y3[capture.py<br/>chapters / transcript / comments]
        Y3 --> Y4[enrich.py<br/>join text to clips]
    end

    subgraph TikTok
        T1[discover.py<br/>Camoufox intercepts<br/>/api/search/item/full/] --> T2[expand.py<br/>yt-dlp account catalogs]
        T2 --> T3[batch.py<br/>yt-dlp mp4]
        T3 --> T4[ffmpeg frames<br/>+ OCR consensus]
        T3 --> T5[ffmpeg wav]
    end

    subgraph Reddit
        R1[scrape.py<br/>old.reddit HTML regex] --> R2[yt-dlp --cookies-from-browser<br/>v.redd.it audio]
        R2 --> R3[title + selftext<br/>+ top comments]
    end

    Y4 --> CORPUS[(corpus.jsonl<br/>unified schema)]
    T4 --> CORPUS
    T5 --> CORPUS
    R3 --> CORPUS
```

---

## 3. Clean + Segment cascade (CPU-first, cheap to expensive)

```mermaid
flowchart TD
    IN[Raw audio WAV] --> E{Energy / RMS<br/>above floor?}
    E -- no --> DROP1[drop: silence]
    E -- yes --> V{Silero VAD<br/>speech?}
    V -- yes --> DROP2[drop: voice / narration]
    V -- no --> F{Spectral flatness<br/>broadband?}
    F -- yes --> DROP3[drop: wind / hiss / static]
    F -- no --> M{CLAP music score<br/>> 0.5?}
    M -- yes --> DROP4[drop: music]
    M -- no --> C{CLAP mechanical<br/>margin >= 0.50?}
    C -- no --> REVIEW[flag for review]
    C -- yes --> L[L1 sound type<br/>+ cyclic / spectral features]
    L --> OUT[clean mechanical<br/>segment clip]
```

This identical cascade runs in **two places**: building the training corpus, and
cleaning a user-uploaded clip at inference. That symmetry is what keeps inference
matched to training.

---

## 4. Label fusion + trust tiering

```mermaid
flowchart LR
    A[Audio signals<br/>L1 type + rhythm] --> FUSE[fusion.py<br/>LLM reasons over signals]
    B[OCR part label] --> FUSE
    C[Chapter / transcript] --> FUSE
    D[Title / comments] --> FUSE

    FUSE --> R{cause + text<br/>corroboration?}
    R -- fault + clean text --> GOLD[gold<br/>trustworthy cause]
    R -- fault + audio only --> SILVER[silver]
    R -- no cause / normal --> BRONZE[bronze]

    GOLD --> MG{music_gate<br/>contaminated?}
    MG -- yes --> EXCL[excluded]
    MG -- no --> KEEP[training set]
    SILVER --> KEEP
```

Rule encoded in `fusion.py`: confidence ≥ 0.7 **requires** corroborating text;
sound-type alone caps at 0.45. Trust is multi-signal agreement, not model
self-confidence.

---

## 5. Training pipeline

```mermaid
flowchart TD
    CORPUS[(tiered corpus.jsonl)] --> PREP[prepare.py<br/>leakage-safe splits<br/>grouped by creator/video]
    PREP --> CAUSES[causes.py<br/>359 raw -> ~24 part families]
    CAUSES --> EMBED[embed.py<br/>frozen CLAP 512-d -> .npz]
    EMBED --> TRAIN[train_best.py<br/>LogReg + StandardScaler heads:<br/>kind / knock / cause]
    TRAIN --> CALIB[confidence.py<br/>isotonic calibration<br/>HIGH/MED/LOW/ABSTAIN]
    CALIB --> EVAL[eval/*<br/>creator-grouped CV<br/>coverage @ precision, ECE]
    CALIB --> MODEL[(best_model_clap.joblib)]
```

---

## 6. Abstracted inference: one core, two front-ends

```mermaid
flowchart TD
    subgraph Frontends
        CLI["CLI<br/>app/cli.py audio.wav"]
        WEB["Local web<br/>app/web.py (FastAPI)<br/>upload .wav"]
    end

    CLI --> CLEAN
    WEB --> UPLOAD[receive upload] --> CLEAN

    subgraph Core["Inference core (shared)"]
        CLEAN["clean.py<br/>same cascade as training:<br/>remove music / voice / noise"]
        CLEAN --> EMB[CLAP embed]
        EMB --> HEADS[linear heads<br/>kind / knock / cause]
        HEADS --> BAND[calibrated<br/>confidence band]
    end

    BAND --> OUT["result JSON<br/>verdict, fault_probability,<br/>engine_knock_probability, top_causes[]"]
    OUT --> CLI
    OUT --> WEB
```

The CLI and the web app are thin shells. Both call the **same** `clean()` then the
**same** `predict()`. Adding a new front-end (or an API) means writing a shell that
calls `predict()` — never re-implementing cleaning or inference.

---

## 7. Sequence: a user uploads a clip to the local web app

```mermaid
sequenceDiagram
    participant U as User (browser)
    participant W as Local web (FastAPI)
    participant CL as clean.py (cascade)
    participant P as predict (CLAP + heads)

    U->>W: POST /diagnose  (clip.wav)
    W->>CL: clean(clip.wav)
    Note over CL: energy → VAD → flatness → music<br/>strip voice / music / silence
    CL-->>W: clean mechanical segment(s)
    W->>P: predict(segment)
    Note over P: CLAP embed → kind/knock/cause heads<br/>→ isotonic calibration
    P-->>W: { verdict, fault_probability,<br/>top_causes[], confidence band }
    W-->>U: result card (honest band, ranked causes)
```
