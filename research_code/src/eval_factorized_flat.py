"""Fail-closed matched evaluation and paired video-cluster bootstrap."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
from src.eval import VideoRecord, temporal_testing_annotations, get_video_labels, calculate_weighted_scores
from src.video_factorized_flat_score import METHODS, read, aggregate
from src import video_score_utils as u

def validate_scores(scores, details, errors, keys, factorized=False):
    if errors or set(scores)!=set(keys) or set(details)!=set(keys):
        raise ValueError(f'Incomplete run: errors={len(errors)}, missing={len(set(keys)-set(scores))}, details={len(details)}')
    for k in keys:
        s=scores[k]; d=details[k]
        if not isinstance(s,(int,float)) or not math.isfinite(s) or not 0<=s<=1 or d.get('score')!=s:
            raise ValueError(f'Invalid score/detail at {k}')
        if factorized:
            score, reconstructed=aggregate(d['candidates'])
            if abs(score-s)>1e-10 or any(not math.isfinite(b[field]) or abs(a[field]-b[field])>1e-10
                for a,b in zip(reconstructed,d['candidates']) for field in ('candidate','margin','weight')):
                raise ValueError('Inconsistent candidate diagnostics')

def metrics(y,s):
    precision,recall,_=precision_recall_curve(y,s)
    return dict(roc_auc=float(roc_auc_score(y,s)),ap=float(average_precision_score(y,s)),
        pr_auc=float(auc(recall,precision)),max_f1=float(np.max(2*precision*recall/(precision+recall+1e-8))))

def paired_bootstrap(labels, predictions, repeats=1000, seed=2026):
    if repeats<1000: raise ValueError('At least 1000 bootstrap repetitions required')
    rng=np.random.default_rng(seed); names=list(predictions); n=len(labels)
    differences={f'{a}_minus_{b}':{'roc_auc':[],'ap':[]} for a,b in
        [(names[0],names[1]),(names[2],names[0]),(names[2],names[1])]}
    skipped=0
    for _ in range(repeats):
        indices=rng.integers(0,n,size=n)
        y=np.concatenate([labels[i] for i in indices])
        if len(np.unique(y))<2:
            skipped+=1; continue
        values={m:metrics(y,np.concatenate([predictions[m][i] for i in indices])) for m in names}
        for a,b in [(names[0],names[1]),(names[2],names[0]),(names[2],names[1])]:
            for metric in ('roc_auc','ap'):
                differences[f'{a}_minus_{b}'][metric].append(values[a][metric]-values[b][metric])
    if repeats-skipped<1000: raise ValueError('Fewer than 1000 valid video bootstrap replicates')
    return dict(seed=seed,repetitions=repeats,valid=repeats-skipped,skipped=skipped,
        confidence_intervals={pair:{metric:[float(x) for x in np.quantile(vals,[.025,.975])]
        for metric,vals in ms.items()} for pair,ms in differences.items()})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('video_dir','index_file','temporal_annotation_file','results_root'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--smoke',action='store_true'); p.add_argument('--bootstrap',type=int,default=1000)
    args=p.parse_args()
    rows=[l.split() for l in args.index_file.read_text().splitlines() if l.strip()]
    temporal_rows=[l.split() for l in args.temporal_annotation_file.read_text().splitlines() if l.strip()]
    names=[u.output_stem(r[0]) for r in rows]
    if len(rows)!=360 or len(set(names))!=360 or len(temporal_rows)!=360 or {u.output_stem(r[0]) for r in temporal_rows}!=set(names):
        raise ValueError('Index/temporal annotations must cover the SAME 360 unique videos')
    if sum(r[3]=='0' for r in rows)!=120:
        raise ValueError('Expected 120 normal videos labeled 0; check index label convention')
    if any(int(r[1])!=0 for r in rows): raise ValueError('Full videos must start at frame zero')
    annotations=temporal_testing_annotations(str(args.temporal_annotation_file))
    if args.smoke: rows=rows[:2]
    labels=[]; predictions={m:[] for m in METHODS}; elapsed={m:0. for m in METHODS}; total=0
    manifests={m:read(args.results_root/m/'_experiment.json') for m in METHODS}
    for m,manifest in manifests.items():
        if (manifest.get('method')!=m or manifest.get('smoke')!=args.smoke or
            manifest.get('names')!=[r[0] for r in rows] or
            manifest.get('video_dir')!=str(args.video_dir.resolve()) or
            manifest.get('index_sha256')!=hashlib.sha256(args.index_file.read_bytes()).hexdigest()):
            raise ValueError(f'Provenance mismatch: {m}')
    shared=['attn_implementation','prompt','model','video_dir','frame_interval','window_seconds','sample_fps','max_frames','precise_time','temperature','source_sha256']
    if any(any(manifests[m][k]!=manifests[METHODS[0]][k] for k in shared) for m in METHODS):
        raise ValueError('Unmatched run configurations')
    for row in rows:
        name=u.output_stem(row[0]); info=u.get_video_info(u.resolve_video_path(args.video_dir,row[0]))
        video=VideoRecord(row,'')
        if info.frame_count!=video.num_frames: raise ValueError(f'Video/annotation frame-count mismatch: {name}')
        windows=list(u.iter_video_windows(info,16,10)); keys=[str(w.center_frame) for w in windows]
        total+=len(keys); reference=None
        for m in METHODS:
            root=args.results_root/m
            scores=read(root/f'{name}.json'); details=read(root/'_details'/f'{name}.json')
            errors=read(root/'_errors'/f'{name}.json')
            validate_scores(scores,details,errors,keys,m==METHODS[0])
            fingerprints=[(details[k]['frames'],details[k]['start_time'],details[k]['end_time']) for k in keys]
            if any(details[k]['start_time']!=w.start_time or details[k]['end_time']!=w.end_time for k,w in zip(keys,windows)):
                raise ValueError('Window boundaries do not match Probe-VAD')
            if reference is not None and reference!=fingerprints: raise ValueError(f'Frame mismatch: {name}/{m}')
            reference=fingerprints
            ordered={k:scores[k] for k in keys}  # old evaluator uses insertion order
            s=calculate_weighted_scores(ordered,16,smooth=True,smoothing_sigma=10)[:video.num_frames]
            if len(s)!=video.num_frames: raise ValueError('No padding permitted')
            predictions[m].append(s)
            elapsed[m]+=sum(details[k]['elapsed_seconds'] for k in keys)
        labels.append(np.asarray(get_video_labels(video,annotations,0))!=0)
    report=dict(status='smoke_passed' if args.smoke else 'complete',videos=len(rows),windows=total,
        inference_failures=0,normal_label=0,smoothing_sigma=10,elapsed_window_seconds=elapsed,
        annotation_sha256=hashlib.sha256(args.temporal_annotation_file.read_bytes()).hexdigest())
    if not args.smoke:
        y=np.concatenate(labels)
        report['metrics']={m:metrics(y,np.concatenate(v)) for m,v in predictions.items()}
        report['paired_bootstrap']=paired_bootstrap(labels,predictions,args.bootstrap)
    u.atomic_write_json(args.results_root/'_report.json',report)
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
