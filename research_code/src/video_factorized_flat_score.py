"""Independent matched three-way ablation; original methods remain unchanged."""
from __future__ import annotations
import argparse, copy, hashlib, json, logging, math, time
from pathlib import Path
import numpy as np
from src import video_score_utils as u
from src.video_cumulative_score import CUMULATIVE_PROMPT
METHODS = ('factorized_flat_11class', 'flat_11class_matched', 'probe_vad_matched')
SCORES = tuple(k / 10 for k in range(11))
QUESTION = ('Considering the allowed scores [0.0], [0.1], [0.2], [0.3], [0.4], [0.5], '
 '[0.6], [0.7], [0.8], [0.9], and [1.0], is [{score:.1f}] the single most '
 'appropriate anomaly severity score for this video segment, based only on '
 'visible evidence? Answer exactly YES or NO.')
def questions():
    return [QUESTION.format(score=s) for s in SCORES]
def aggregate(details):
    if len(details) != 11:
        raise ValueError('Exactly eleven results required')
    if not all(math.isfinite(float(d[k])) for d in details for k in ('yes_loglik','no_loglik')):
        raise ValueError('Non-finite likelihood')
    margins = [float(d['yes_loglik'])-float(d['no_loglik']) for d in details]
    exps = [math.exp(m-max(margins)) for m in margins]
    weights = [x/sum(exps) for x in exps]  # fixed T=1, no PAVA
    score = sum(s*w for s,w in zip(SCORES,weights))
    return score, [dict(candidate=s, yes_loglik=float(d['yes_loglik']),
        no_loglik=float(d['no_loglik']), margin=m, weight=w)
        for s,d,m,w in zip(SCORES,details,margins,weights)]
def tokenizer_report(processor):
    ids = {s:list(processor.tokenizer(s,add_special_tokens=False).input_ids) for s in ('YES','NO')}
    if any(not v for v in ids.values()) or ids['YES']==ids['NO']:
        raise ValueError(f'Invalid tokenization: {ids}')
    return ids

def score_loaded(args, model, processor, loaded):
    if args.method == METHODS[1]:
        score,raw,error = u.score_conversation(model,processor,loaded,
            'likelihood_optimized',0.5,12,'NORMAL','ANOMALOUS',True,1.0)
        if error: raise RuntimeError(error)
        return score, {'raw':raw}
    texts = questions() if args.method==METHODS[0] else [
        u.build_threshold_conversation(Path('unused'),u.VideoWindow(0,0,1),2,10,args.prompt,t,True)[1]['content'][1]['text']
        for t in u.CUMULATIVE_THRESHOLDS]
    conversations=[]
    for text in texts:
        conv=copy.deepcopy(loaded)
        conv[1]['content'][1]['text']=text
        conversations.append(conv)
    tokens=tokenizer_report(processor)
    if all(len(v)==1 for v in tokens.values()):
        probabilities,details=u.cumulative_threshold_likelihood_optimized(model,processor,
            conversations,temperature=1.0,threshold_batch_size=args.threshold_batch_size,prefix_cache=True)
    else:
        from src.factorized_sequence_likelihood import sequence_binary_likelihood
        probabilities,details=sequence_binary_likelihood(model,processor,conversations,
            threshold_batch_size=args.threshold_batch_size)
    if args.method==METHODS[0]:
        score,candidates=aggregate(details)
        return score, {'candidates':candidates,'answer_token_ids':tokens}
    score,projected,violations=u.cumulative_score_from_tail_probabilities(probabilities,monotonic_projection=True)
    return score, {'thresholds':details,'projected':projected,'violations':violations}

def read(path):
    if not path.exists(): return {}
    value=json.loads(path.read_text())
    if not isinstance(value,dict): raise ValueError(f'Invalid JSON: {path}')
    return value

def frame_fingerprint(loaded):
    item=loaded[1]['content'][0]; digest=hashlib.sha256()
    for frame in item['video']:
        a=np.asarray(frame); digest.update(str((a.shape,str(a.dtype))).encode()); digest.update(a.tobytes())
    timestamps=[float(t) for t in item['timestamps']]
    digest.update(json.dumps(timestamps).encode())
    return dict(sha256=digest.hexdigest(),timestamps=timestamps,num_frames=len(item['video']))

def process_video(args,model,processor,name):
    path=u.resolve_video_path(args.video_dir,name)
    windows=list(u.iter_video_windows(u.get_video_info(path),16,10))
    stem=u.output_stem(name); target=args.output_dir/f'{stem}.json'
    dp=args.output_dir/'_details'/f'{stem}.json'; ep=args.output_dir/'_errors'/f'{stem}.json'
    scores,details,errors=read(target),read(dp),read(ep)
    dirty=int('__video__' in errors)
    errors.pop('__video__',None)
    def checkpoint():
        nonlocal dirty
        if dirty:
            u.atomic_write_json(dp,details); u.atomic_write_json(target,scores); u.atomic_write_json(ep,errors)
            dirty=0
    try:
        for window in windows:
            key=str(window.center_frame)
            if key in scores and key in details and key not in errors:
                if math.isfinite(scores[key]) and 0<=scores[key]<=1 and scores[key]==details[key].get('score'):
                    continue
            start=time.perf_counter()
            try:
                conv=u.build_video_conversation(path,window,2,10,args.prompt,'likelihood_optimized',True)
                loaded=u.preload_video_conversation(processor,conv,path,window,2,10,True)
                fingerprint=frame_fingerprint(loaded)
                score,detail=score_loaded(args,model,processor,loaded)
                if not math.isfinite(score) or not 0<=score<=1: raise ValueError(f'Invalid score {score}')
                details[key]=dict(detail,score=score,frames=fingerprint,start_time=window.start_time,
                    end_time=window.end_time,elapsed_seconds=time.perf_counter()-start)
                scores[key]=score; errors.pop(key,None)
            except Exception as exc:
                scores.pop(key,None); details.pop(key,None)
                errors[key]=dict(error=repr(exc),start_time=window.start_time,end_time=window.end_time)
                logging.exception('%s window %s failed',name,key)
            dirty+=1
            if dirty>=args.checkpoint_interval: checkpoint()
    finally: checkpoint()
    return len(errors)

def guard_manifest(output,manifest):
    output.mkdir(parents=True,exist_ok=True); path=output/'_experiment.json'
    if path.exists():
        if read(path)!=manifest: raise RuntimeError('Resume configuration differs; use NEW directory')
    else:
        if any(output.iterdir()): raise RuntimeError('Refusing unowned/nonempty output directory')
        u.atomic_write_json(path,manifest)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('video_dir','index_file','output_dir'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model_path',default=u.DEFAULT_MODEL_PATH)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--attn_implementation',choices=['flash_attention_2','eager'],default='flash_attention_2')
    p.add_argument('--method',choices=METHODS,default=METHODS[0])
    p.add_argument('--prompt',default=CUMULATIVE_PROMPT)
    p.add_argument('--threshold_batch_size',type=int,default=11)
    p.add_argument('--checkpoint_interval',type=int,default=20)
    p.add_argument('--smoke',action='store_true',help='First two COMPLETE videos, separate output')
    args=p.parse_args()
    if args.prompt!=CUMULATIVE_PROMPT or min(args.threshold_batch_size,args.checkpoint_interval)<1:
        p.error('Require shared CUMULATIVE_PROMPT and positive batch/checkpoint sizes')
    from src.factorized_cuda_preflight import check
    runtime=check(args.attn_implementation,args.device)
    names=u.load_video_names(args.video_dir,args.index_file)
    if len(names)!=360 or len({u.output_stem(n) for n in names})!=360: p.error('Require unique 360-video index')
    if args.smoke: names=names[:2]
    manifest=dict(schema=1,method=args.method,prompt=args.prompt,question=QUESTION,model=args.model_path,
        video_dir=str(args.video_dir.resolve()),index_sha256=hashlib.sha256(args.index_file.read_bytes()).hexdigest(),
        names=names,frame_interval=16,window_seconds=10,sample_fps=2,max_frames=10,precise_time=True,
        temperature=1.0,smoke=args.smoke,attn_implementation=args.attn_implementation,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()+Path(u.__file__).read_bytes()+Path(__file__).with_name('factorized_sequence_likelihood.py').read_bytes()).hexdigest())
    guard_manifest(args.output_dir,manifest)
    import fcntl
    with (args.output_dir/'_run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        u.atomic_write_json(args.output_dir/'_runtime.json',runtime)
        model,processor=u.load_model(args.model_path,args.device,args.attn_implementation)
        u.atomic_write_json(args.output_dir/'_tokenizer.json',tokenizer_report(processor))
        failures=0
        for name in names:
            try: failures+=process_video(args,model,processor,name)
            except Exception as exc:
                u.atomic_write_json(args.output_dir/'_errors'/f'{u.output_stem(name)}.json',{'__video__':{'error':repr(exc)}})
                failures+=1
        print(json.dumps(dict(videos=len(names),failures=failures,smoke=args.smoke)))
        if failures: raise SystemExit(2)
if __name__=='__main__': main()
