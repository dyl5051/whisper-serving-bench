# v1 Decision Matrix

## NVIDIA L4

### Aggregate RTF (lower is faster)

| framework \ concurrency | 1 | 8 |
|---|---|---|
| faster_whisper | 0.1673 | 0.1751 |


### Latency p95 (seconds)

| framework \ concurrency | 1 | 8 |
|---|---|---|
| faster_whisper | 19.862 | 107.121 |


### WER

| framework \ concurrency | 1 | 8 |
|---|---|---|
| faster_whisper | 0.0403 | 0.0368 |


### GPU util mean (%)

| framework \ concurrency | 1 | 8 |
|---|---|---|
| faster_whisper | 94.3 | 93.8 |


### Cost USD/audio-hour (on-demand)

| framework \ concurrency | 1 | 8 |
|---|---|---|
| faster_whisper | $0.1004 | $0.1050 |

