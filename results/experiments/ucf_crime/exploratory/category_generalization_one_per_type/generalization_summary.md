# Modify2 one-video-per-category generalization check

Run date: 2026-07-17

Selection rule: for each of the 13 UCF-Crime anomaly categories and the Normal category, choose the shortest test video that has all four required inputs: raw video, author caption, author refined score, and temporal annotation. Selection does not use model performance. Normal is included only in pooled metrics and false-positive inspection because a normal-only video has no defined per-video ROC-AUC or AP.

Total: 14 videos, 13,936 frames, 2,999 positive frames. No visualization was generated.

## Pooled micro metrics over all 14 videos

| Method | ROC-AUC | AP |
|---|---:|---:|
| Author | 0.611429 | 0.293888 |
| Author caption fixed + Modify2 | 0.627656 | 0.300392 |
| Modify2, 2 s at 2 FPS | **0.676543** | **0.377332** |
| Modify2, 2 s at 5 FPS | 0.666817 | 0.355196 |

## Macro average over 13 anomaly videos

| Method | ROC-AUC | AP |
|---|---:|---:|
| Author | 0.586076 | 0.353015 |
| Author caption fixed + Modify2 | 0.609293 | 0.437475 |
| Modify2, 2 s at 2 FPS | **0.747587** | **0.564684** |
| Modify2, 2 s at 5 FPS | 0.699289 | 0.516221 |

## Per-category metrics

| Category | Author ROC | Fixed ROC | FPS2 ROC | FPS5 ROC | Author AP | Fixed AP | FPS2 AP | FPS5 AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Abuse | 0.531437 | 0.484045 | 0.649779 | **0.709502** | 0.086364 | 0.060087 | 0.088152 | **0.287139** |
| Arrest | 0.966442 | **0.991302** | 0.844776 | 0.966583 | 0.694528 | **0.925289** | 0.770113 | 0.894088 |
| Arson | 0.627512 | 0.368882 | **0.908295** | 0.829985 | 0.187925 | 0.111787 | **0.458118** | 0.352431 |
| Assault | 0.557792 | **0.695570** | 0.651967 | 0.563607 | 0.300801 | 0.383774 | **0.516923** | 0.427919 |
| Burglary | 0.667967 | 0.910663 | **0.923405** | 0.711370 | 0.415729 | **0.824273** | 0.788547 | 0.418991 |
| Explosion | 0.376995 | 0.369556 | 0.520719 | **0.892758** | 0.322325 | 0.272617 | 0.320302 | **0.729854** |
| Fighting | 0.748058 | 0.756305 | **0.906596** | 0.849012 | 0.420213 | 0.380788 | **0.760856** | 0.623961 |
| RoadAccidents | 0.587984 | **0.851500** | 0.678497 | 0.070370 | 0.552618 | **0.826332** | 0.518272 | 0.281115 |
| Robbery | 0.365599 | 0.211247 | 0.989189 | **0.990541** | 0.084693 | 0.070404 | **0.930379** | 0.919299 |
| Shooting | 0.521649 | 0.133646 | **0.888632** | 0.853418 | 0.417084 | 0.278943 | **0.749287** | 0.660530 |
| Shoplifting | 0.444945 | **0.887792** | 0.611846 | 0.466875 | 0.443042 | **0.786633** | 0.688375 | 0.435523 |
| Stealing | 0.529377 | 0.590268 | 0.618445 | **0.671603** | 0.465594 | **0.564451** | 0.618235 | 0.548572 |
| Vandalism | **0.693236** | 0.670029 | 0.526490 | 0.515130 | 0.198274 | **0.201793** | 0.133327 | 0.131451 |

## Win counts and observations

- Author caption fixed + Modify2 beats Author on 7/13 categories in ROC and 7/13 in AP.
- FPS2 Modify2 beats Author on 11/13 categories in ROC and 10/13 in AP.
- FPS5 Modify2 beats Author on 11/13 categories in ROC and 10/13 in AP.
- FPS5 beats FPS2 on only 5/13 categories in ROC and 3/13 in AP.
- The FPS2 variant is the strongest pooled and macro-average configuration in this check.
- FPS5 helps Abuse, Explosion, and some other fast/localized events, but severely hurts the very short RoadAccidents sample and is not a uniformly better sampling rate.
- Both regenerated-caption variants underperform Author on Vandalism, so the remaining failure is not solved by higher short-window sampling.
- On the selected Normal video, mean/max scores are: fixed 0.077849/0.120776, FPS2 0.110505/0.158330, FPS5 0.175802/0.203461. FPS5 raises the normal-video score floor.

## Integrity checks

- The fixed-caption variant matches 879/879 author captions exactly.
- Every variant has 14 final score files and every video has 15 candidate review entries.
- The inference output tree contains no MP4, AVI, or GIF.
- Author refined scores and ground truth are only introduced by the final evaluation command.

This is a preliminary category-coverage check, not a full generalization claim: one short video per category is a small and duration-biased sample. Full-test evaluation or repeated random samples per category are still required for a publication-grade conclusion.
