# MSAD Prompt Sensitivity

This experiment keeps the model, visual sampling, system instruction,
thresholds, and decoding settings fixed while changing only the threshold
question. All results use the complete 360-video MSAD protocol.

Metrics below are percentages.

| Prompt ID | ROC-AUC | PR-AUC | Max-F1 |
|---|---:|---:|---:|
| `visible_evidence` | 87.2998 | 78.2773 | 75.5576 |
| `reach_level` | 86.9473 | 78.6185 | 74.7221 |
| `no_less_than` | 86.9702 | 78.4240 | 74.9077 |
| `rated_above` | 87.7588 | 78.9726 | 75.2856 |

The strongest ROC-AUC and PR-AUC in this comparison come from
`rated_above`. The result is a prompt-sensitivity observation, not a claim
that the prompt should be selected on the test set.

## Question templates

```text
visible_evidence:
Considering only what is visibly shown in this video segment, is its anomaly
severity at least {threshold:.1f} on the defined scale? Answer exactly YES or NO.

reach_level:
Does the visible activity in this video segment reach severity level
{threshold:.1f} or higher on the defined anomaly scale? Answer exactly YES or NO.

no_less_than:
Using only the current clip's visual evidence, is this segment no less
anomalous than severity {threshold:.1f} on the defined severity scale? Answer
exactly YES or NO.

rated_above:
Would the anomaly visible in this video segment be rated {threshold:.1f} or
above on the defined severity scale? Answer exactly YES or NO.
```
