# Dataset and annotation preparation

COVAS-VAD does not redistribute dataset media. The release includes copies of
the exact evaluation index and temporal annotation text used for the bundled
results; those files remain subject to their upstream licenses. Obtain
UCF-Crime, MSAD, and XD-Violence videos from their official sources.

## Directory layout

```text
data/<dataset>/
├── videos/
│   ├── <video_id>.mp4
│   └── ...
├── frames/
└── annotations/
    ├── anomaly_test.txt
    └── temporal_annotations.txt
```

Supported video extensions are `.mp4`, `.avi`, `.mov`, and `.mkv`. Index
identifiers may include or omit the extension. Identifiers containing dots,
such as XD-Violence movie names, are preserved rather than truncated at the
last dot.

## Test index format

Each non-empty line contains:

```text
video_id start_frame end_frame class_label
```

Example:

```text
Abuse028_x264 0 1411 0
Normal_Videos_001_x264 0 719 7
```

- `start_frame` and `end_frame` are inclusive.
- `class_label` may contain comma-separated labels for multiple anomaly
  intervals.
- The first field is also used to locate the raw video and name its score JSON.

## Temporal annotation format

Each line contains:

```text
video_id event_name start_1 end_1 start_2 end_2 ... -1 -1
```

Example:

```text
Abuse028_x264 Abuse 165 240 -1 -1
Normal_Videos_001_x264 Normal -1 -1
```

Intervals use original-frame indices and inclusive boundaries. `-1` entries
are ignored.

## Configuration templates

- `configs/ucf_crime.env.example`
- `configs/msad.env.example`
- `configs/xd_violence.env.example`

Copy a template, adjust all paths, and run:

```bash
bash scripts/run_from_config.sh configs/<dataset>.env
```

The historical normal-label conventions used in this project are:

| Dataset | Normal label | Precise time |
|---|---:|---:|
| UCF-Crime | 7 | 0 |
| MSAD | 0 | 1 |
| XD-Violence | 4 | 0 |

Always verify these values against the annotation files you actually use.
Different repackagings of a dataset may use different class IDs or video
subsets.

## Validation checklist

Before a full run:

1. Every first-column index identifier resolves to exactly one video.
2. Video FPS and frame counts are nonzero.
3. Index and temporal files use matching basenames.
4. `end_frame - start_frame + 1` matches the intended evaluation length.
5. The normal label matches the index file.
6. MSAD uses `PRECISE_TIME=1` if videos have nonzero stream start timestamps.
