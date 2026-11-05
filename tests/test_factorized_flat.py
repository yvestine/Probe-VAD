"""CPU protocol tests with deterministic stand-in model; not VLM results."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research_code'))
from src import video_factorized_flat_score as f
from src import video_score_utils as u
from src.factorized_sequence_likelihood import sequence_binary_likelihood
from src.eval_factorized_flat import validate_scores, paired_bootstrap

class Processor:
    def __init__(self,multi=False): self.multi=multi; self.loads=0
    def tokenizer(self,text,**kwargs):
        return NS(input_ids=([1,2] if self.multi else [1]) if text=='YES' else [3])
    def _load_multimodal_data(self,c): self.loads+=1; return c
    def _gather_multimodal_data(self,c): return [0]
    def process_images(self,*a,**kw):
        return dict(pixel_values=torch.zeros(1),grid_sizes=torch.ones((1,1),dtype=torch.long),
                    merge_sizes=torch.ones(1,dtype=torch.long),modals=['video'])
    def _process_conversation_without_label(self,c,*a,**kw):
        import re
        text=c[1]['content'][1]['text']
        match=re.search(r'is \[(\d\.\d)\]',text)
        i=round(float(match[1])*10) if match else 0
        return dict(input_ids=torch.tensor([[7,8,10+i]]),attention_mask=torch.ones((1,3),dtype=torch.long))

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.embed_tokens=torch.nn.Embedding(32,4); self.encodes=0
        self.config=NS(use_token_compression=False,image_token_index=99)
    def get_model(self): return self
    def encode_images(self,*a): self.encodes+=1; return torch.empty((0,4))
    def _get_valid_visual_tokens(self,features,*a): return features
    def _get_compression_mask(self,*a): return torch.empty(0)
    def _maybe_truncate_visual_tokens(self,features,mask,*a): return features,mask
    def forward(self,inputs_embeds,**kw):
        return NS(logits=torch.arange(32,dtype=torch.float32).div(10).expand(*inputs_embeds.shape[:2],32))

def conversations():
    return [[{'role':'system','content':f.CUMULATIVE_PROMPT}, {'role':'user','content':[
        {'type':'video','video':[np.zeros((2,2,3),dtype=np.uint8)],'timestamps':[0.], 'num_frames':1},
        {'type':'text','text':q}]}] for q in f.questions()]

def test_questions():
    qs=f.questions(); assert len(set(qs))==11
    for i,q in enumerate(qs):
        assert q.replace(f'is [{i/10:.1f}]','is [SCORE]')==qs[0].replace('is [0.0]','is [SCORE]')

def test_optimized_handles_eleven_and_one_vision_encode():
    m,p=Model(),Processor()
    probs,details=u.cumulative_threshold_likelihood_optimized(m,p,conversations(),threshold_batch_size=4,prefix_cache=False)
    assert len(details)==len(probs)==11 and m.encodes==p.loads==1
    ll=torch.log_softmax(torch.arange(32).float()/10,dim=0)
    assert details[-1]['yes_loglik']==pytest.approx(ll[1].item())

def test_complete_multitoken_continuation_and_shared_vision():
    m,p=Model(),Processor(True)
    _,ds=sequence_binary_likelihood(m,p,conversations(),threshold_batch_size=4)
    ll=torch.log_softmax(torch.arange(32).float()/10,dim=0)
    assert len(ds)==11 and m.encodes==p.loads==1
    assert ds[-1]['yes_loglik']==pytest.approx((ll[1]+ll[2]).item())
    assert ds[-1]['no_loglik']==pytest.approx(ll[3].item())

@pytest.mark.parametrize('peak',[0,5,10])
def test_stable_weights_and_score(peak):
    score,ds=f.aggregate([dict(yes_loglik=0 if i==peak else -10000,no_loglik=-2) for i in range(11)])
    assert score==pytest.approx(peak/10) and sum(d['weight'] for d in ds)==pytest.approx(1)
    assert 0<=score<=1

def test_invalid_likelihoods():
    with pytest.raises(ValueError): f.aggregate([dict(yes_loglik=float('nan'),no_loglik=0)]*11)

def test_output_guard(tmp_path):
    f.guard_manifest(tmp_path,{'a':1}); f.guard_manifest(tmp_path,{'a':1})
    with pytest.raises(RuntimeError): f.guard_manifest(tmp_path,{'a':2})
    other=tmp_path/'history'; other.mkdir(); (other/'old.json').write_text('{}')
    with pytest.raises(RuntimeError): f.guard_manifest(other,{'a':1})

def test_fail_closed():
    with pytest.raises(ValueError): validate_scores({'0':.5},{},{'0':{}},['0'])
    with pytest.raises(ValueError): validate_scores({'0':.5},{'0':{'score':.5}},{},['0','16'])

def test_interruption_resume_and_retry(tmp_path):
    args=NS(video_dir=tmp_path,output_dir=tmp_path/'out',prompt=f.CUMULATIVE_PROMPT,checkpoint_interval=20)
    info=u.VideoInfo(10.,33,3.3); loaded=conversations()[0]
    detail={'candidates':[dict(yes_loglik=-1,no_loglik=-1)]*11}
    calls=[]
    def score(*a):
        calls.append(1)
        if len(calls)==2: raise KeyboardInterrupt
        return .5,detail
    with patch.object(u,'resolve_video_path',return_value=tmp_path/'test.avi'),patch.object(u,'get_video_info',return_value=info),patch.object(u,'preload_video_conversation',return_value=loaded),patch.object(f,'score_loaded',side_effect=score):
        with pytest.raises(KeyboardInterrupt): f.process_video(args,None,None,'test')
    assert set(f.read(args.output_dir/'test.json'))=={'0'}
    with patch.object(u,'resolve_video_path',return_value=tmp_path/'test.avi'),patch.object(u,'get_video_info',return_value=info),patch.object(u,'preload_video_conversation',return_value=loaded),patch.object(f,'score_loaded',side_effect=[RuntimeError('test failure'),(.5,detail)]) as mock:
        assert f.process_video(args,None,None,'test')==1 and mock.call_count==2
    assert '16' not in f.read(args.output_dir/'test.json')
    with patch.object(u,'resolve_video_path',return_value=tmp_path/'test.avi'),patch.object(u,'get_video_info',return_value=info),patch.object(u,'preload_video_conversation',return_value=loaded),patch.object(f,'score_loaded',return_value=(.5,detail)) as mock:
        assert f.process_video(args,None,None,'test')==0 and mock.call_count==1
    assert set(f.read(args.output_dir/'test.json'))=={'0','16','32'}

def test_paired_bootstrap_identical_is_zero():
    ys=[np.array([0,1]),np.array([0,1])]; ss=[np.array([.1,.8]),np.array([.4,.6])]
    r=paired_bootstrap(ys,{m:ss for m in f.METHODS},1000)
    assert r['valid']==1000
    assert all(ci==[0.,0.] for ms in r['confidence_intervals'].values() for ci in ms.values())

def test_two_video_synthetic_pipeline_smoke(tmp_path):
    import cv2
    class VideoProcessor(Processor):
        def load_video(self,video_path,start_time,end_time,fps,max_frames,precise_time):
            cap=cv2.VideoCapture(video_path); native=cap.get(cv2.CAP_PROP_FPS)
            frames=[]; times=[]; i=0
            while True:
                ok,frame=cap.read()
                if not ok: break
                t=i/native
                if start_time<=t<end_time and i%5==0:
                    frames.append(frame); times.append(t)
                i+=1
            cap.release()
            return frames,times
    args=NS(video_dir=tmp_path,output_dir=tmp_path/'scores',prompt=f.CUMULATIVE_PROMPT,
        checkpoint_interval=1,threshold_batch_size=4,method=f.METHODS[0])
    model=Model(); processor=VideoProcessor(True)
    for name in ('smoke_a','smoke_b'):
        writer=cv2.VideoWriter(str(tmp_path/f'{name}.avi'),cv2.VideoWriter_fourcc(*'MJPG'),10,(16,16))
        assert writer.isOpened()
        for i in range(17): writer.write(np.full((16,16,3),i*10,dtype=np.uint8))
        writer.release()
        assert f.process_video(args,model,processor,name)==0
        scores=f.read(args.output_dir/f'{name}.json'); ds=f.read(args.output_dir/'_details'/f'{name}.json')
        validate_scores(scores,ds,{},['0','16'],True)
        assert all(len(d['frames']['sha256'])==64 for d in ds.values())
        before=model.encodes
        assert f.process_video(args,model,processor,name)==0 and model.encodes==before
    assert model.encodes==4  # one encode per window, not 11

def test_real_transformers_prefix_cache_matches_uncached_eleven():
    # Exercises actual Transformers 4.46.3 Llama/DynamicCache APIs on CPU.
    from transformers import LlamaConfig, LlamaForCausalLM
    class CachedModel(Model):
        def __init__(self):
            super().__init__()
            config=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
                num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)
            config._attn_implementation='eager'
            self.lm=LlamaForCausalLM(config).eval()
        def encode_images(self,*a): self.encodes+=1; return torch.empty((0,16))
        def get_model(self): return self.lm.model
        def forward(self,**kw): return self.lm(**kw)
    torch.manual_seed(42)
    m=CachedModel().eval()
    _,cached=u.cumulative_threshold_likelihood_optimized(m,Processor(),conversations(),
        threshold_batch_size=4,prefix_cache=True)
    _,uncached=u.cumulative_threshold_likelihood_optimized(m,Processor(),conversations(),
        threshold_batch_size=4,prefix_cache=False)
    assert len(cached)==11
    for a,b in zip(cached,uncached):
        assert a['yes_loglik']==pytest.approx(b['yes_loglik'],abs=2e-6)
        assert a['no_loglik']==pytest.approx(b['no_loglik'],abs=2e-6)

def test_cuda_version_contract():
    from src.factorized_cuda_preflight import PINNED,version_errors
    assert version_errors(dict(PINNED,torch='2.5.1+cu124'),(3,10))==[]
    assert version_errors(dict(PINNED,torch='2.12.1'),(3,14))
    assert version_errors({},(3,10))
