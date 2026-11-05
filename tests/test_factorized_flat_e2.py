"""E2-based flat class tests; synthetic data/model is not real MSAD evidence."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import cv2
import numpy as np
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'research_code'))
from src import video_factorized_flat_e2_score as f
from src import video_ab_calibrated_score as e2
from src import video_score_utils as u

class Model(torch.nn.Module):
    """Deterministic CPU stand-in; no real VLM performance is simulated."""
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(32, 4)
        self.encodes = 0
        self.config = NS(use_token_compression=False, image_token_index=99)

    def get_model(self): return self
    def encode_images(self, *args):
        self.encodes += 1
        return torch.empty((0, 4))
    def _get_valid_visual_tokens(self, features, *args): return features
    def _get_compression_mask(self, *args): return torch.empty(0)
    def _maybe_truncate_visual_tokens(self, features, mask, *args): return features, mask
    def forward(self, inputs_embeds, **kwargs):
        logits = torch.arange(32, dtype=torch.float32).div(10)
        return NS(logits=logits.expand(*inputs_embeds.shape[:2], 32))



def prompts(swapped=False):
    return [f.build_ab_threshold_conversation(Path('v.mp4'),u.VideoWindow(16,0.,2.),2,10,
        e2.CUMULATIVE_PROMPT,c,swapped=swapped,precise_time=True) for c in f.CANDIDATE_SCORES]


def test_eleven_questions_and_exact_option_swap():
    fw,sw=prompts(),prompts(True)
    assert len(fw)==len(sw)==11 and f.CANDIDATE_SCORES==tuple(i/10 for i in range(11))
    templates=[]
    for i,(a,b) in enumerate(zip(fw,sw)):
        t=a[1]['content'][1]['text']; r=b[1]['content'][1]['text']
        lines=t.splitlines(); reverse=r.splitlines()
        assert lines[0]==reverse[0]
        assert lines[1][3:]==reverse[2][3:] and lines[2][3:]==reverse[1][3:]
        assert 'at least' not in t and 'strictly below' not in t
        templates.append(t.replace(f'[{i/10:.1f}]','[SCORE]'))
        old=e2.build_ab_threshold_conversation(Path('v.mp4'),u.VideoWindow(16,0.,2.),2,10,
            e2.CUMULATIVE_PROMPT,i/10,swapped=False,precise_time=True)
        assert a[0]==old[0] and a[1]['content'][0]==old[1]['content'][0]
    assert len(set(templates))==1


def make_logits(margins,bias=0.):
    # Fixed A-token prior cancels after option reversal.
    fw=[dict(a_logit=-2000+m/2+bias,b_logit=-2000-m/2) for m in margins]
    sw=[dict(a_logit=-2000-m/2+bias,b_logit=-2000+m/2) for m in margins]
    return fw+sw


def test_e2_margin_calibration_not_sigmoid_aggregation():
    margins=[-2+i*.7 for i in range(11)]
    score,ds=f.aggregate_calibrated_logits(make_logits(margins,bias=3))
    masses=np.exp(np.asarray(margins)-max(margins)); weights=masses/masses.sum()
    assert [d['calibrated_margin'] for d in ds]==pytest.approx(margins)
    assert [d['class_weight'] for d in ds]==pytest.approx(weights)
    assert score==pytest.approx(np.dot(np.arange(11)/10,weights))
    for raw1,raw2,d in zip(make_logits(margins)[:11],make_logits(margins)[11:],ds):
        _,m,_=e2.calibrated_margin_probability(raw1['a_logit'],raw1['b_logit'],
            swapped_a_logit=raw2['a_logit'],swapped_b_logit=raw2['b_logit'])
        assert m==pytest.approx(d['calibrated_margin'])
    assert not np.allclose(weights,torch.softmax(torch.sigmoid(torch.tensor(margins)),0).numpy())


@pytest.mark.parametrize('peak',[0,5,10])
def test_extreme_margins_and_bounds(peak):
    score,ds=f.aggregate_calibrated_logits(make_logits([1000 if i==peak else -1000 for i in range(11)]))
    assert score==pytest.approx(peak/10)
    assert sum(d['class_weight'] for d in ds)==pytest.approx(1)
    assert 0<=score<=1


def test_invalid_inputs():
    with pytest.raises(ValueError): f.aggregate_calibrated_logits(make_logits([0]*10))
    with pytest.raises(ValueError): f.aggregate_calibrated_logits(make_logits([0]*11),.5)
    with pytest.raises(ValueError): f.aggregate_calibrated_logits(make_logits([float('nan')]*11))


class ABProcessor:
    def __init__(self): self.loads = 0
    def tokenizer(self, text, **kwargs): return NS(input_ids=[1] if text == 'A' else [3])
    def _load_multimodal_data(self, conversation):
        self.loads += 1
        return conversation
    def _gather_multimodal_data(self, conversation): return [0]
    def process_images(self, *args, **kwargs):
        return dict(pixel_values=torch.zeros(1), grid_sizes=torch.ones((1, 1), dtype=torch.long),
                    merge_sizes=torch.ones(1, dtype=torch.long), modals=['video'])
    def _process_conversation_without_label(self, conversation, *args, **kwargs):
        return dict(input_ids=torch.tensor([[7, 8, 10]]),
                    attention_mask=torch.ones((1, 3), dtype=torch.long))


def test_22_prompts_one_visual_encode():
    m,p=Model(),ABProcessor()
    assert e2.validate_single_token_options(p)==(1,3)
    score,ds=f.option_calibrated_flat_score(m,p,prompts(),prompts(True),
        threshold_batch_size=10,prefix_cache=False)
    assert m.encodes==p.loads==1
    assert len(ds)==11 and score==pytest.approx(.5)
    assert all(d['calibrated_margin']==pytest.approx(0) for d in ds)


def args_for(tmp_path):
    return NS(video_dir=tmp_path,output_dir=tmp_path/'out',frame_interval=16,window_seconds=10,
        sample_fps=2,max_frames=10,prompt=e2.CUMULATIVE_PROMPT,a_candidate='A',b_candidate='B',
        precise_time=True,likelihood_temperature=1,optimized=True,threshold_batch_size=10,
        prefix_cache=False,checkpoint_interval=20)


def test_resume_interruption_failure_and_sorted_output(tmp_path):
    args=args_for(tmp_path); result=f.aggregate_calibrated_logits(make_logits([0]*11))
    info=u.VideoInfo(10,33,3.3)
    with patch.object(u,'resolve_video_path',return_value=tmp_path/'v.mp4'),patch.object(u,'get_video_info',return_value=info):
        with patch.object(f,'option_calibrated_flat_score',side_effect=[result,KeyboardInterrupt]):
            with pytest.raises(KeyboardInterrupt): f.process_video(args,None,None,'v')
        assert list(f.read_json(args.output_dir/'v.json'))==['0']
        with patch.object(f,'option_calibrated_flat_score',side_effect=[RuntimeError('injected'),result]):
            assert f.process_video(args,None,None,'v')==1
        with pytest.raises(RuntimeError): f.validate_outputs(args,['v'])
        with patch.object(f,'option_calibrated_flat_score',return_value=result) as call:
            assert f.process_video(args,None,None,'v')==0 and call.call_count==1
        assert list(f.read_json(args.output_dir/'v.json'))==['0','16','32']
        assert f.validate_outputs(args,['v'])['complete']


def test_guard_no_overwrite(tmp_path):
    f.guard_output(tmp_path,{'experiment':f.EXPERIMENT})
    f.guard_output(tmp_path,{'experiment':f.EXPERIMENT})
    with pytest.raises(RuntimeError): f.guard_output(tmp_path,{'experiment':'e2'})
    other=tmp_path/'historical';other.mkdir();(other/'v.json').write_text('{}')
    with pytest.raises(RuntimeError): f.guard_output(other,{'experiment':f.EXPERIMENT})


def test_two_generated_video_smoke(tmp_path):
    class VideoProcessor(ABProcessor):
        def _load_multimodal_data(self,conv):
            self.loads+=1
            v=conv[1]['content'][0]['video'];cap=cv2.VideoCapture(v['video_path'])
            fps=cap.get(cv2.CAP_PROP_FPS); frames=[]; ts=[]; i=0
            while True:
                ok,frame=cap.read()
                if not ok: break
                t=i/fps
                if v['start_time']<=t<v['end_time'] and i%5==0:
                    frames.append(frame);ts.append(t)
                i+=1
            cap.release(); assert frames
            result=copy.deepcopy(conv);item=result[1]['content'][0]
            item.update(video=frames,num_frames=len(frames),timestamps=ts)
            return result
    args=args_for(tmp_path);m,p=Model(),VideoProcessor()
    for name in ['a','b']:
        out=cv2.VideoWriter(str(tmp_path/f'{name}.avi'),cv2.VideoWriter_fourcc(*'MJPG'),10,(16,16))
        assert out.isOpened()
        for i in range(17): out.write(np.full((16,16,3),i*10,np.uint8))
        out.release()
        assert f.process_video(args,m,p,name)==0
        before=m.encodes
        assert f.process_video(args,m,p,name)==0 and m.encodes==before
    report=f.validate_outputs(args,['a','b'])
    assert report['complete'] and report['expected_windows']==4 and m.encodes==p.loads==4

def test_22_query_real_cache_matches_uncached():
    import re
    from transformers import Qwen2Config, Qwen2ForCausalLM
    class VariedProcessor(ABProcessor):
        def _process_conversation_without_label(self,c,*args,**kwargs):
            text=c[1]['content'][1]['text']
            k=round(float(re.search(r'Is \[(\d\.\d)\]',text)[1])*10)
            swap=int(' is not ' in text.splitlines()[1])
            return dict(input_ids=torch.tensor([[7,8,10+k,5+swap]]),
                        attention_mask=torch.ones((1,4),dtype=torch.long))
    class CachedModel(Model):
        def __init__(self):
            super().__init__()
            config=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
                num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)
            config._attn_implementation='eager'
            self.lm=Qwen2ForCausalLM(config).eval(); self.cached_calls=0
        def encode_images(self,*args): self.encodes+=1; return torch.empty((0,16))
        def get_model(self): return self.lm.model
        def forward(self,**kwargs):
            self.cached_calls+=int(kwargs.get('past_key_values') is not None)
            return self.lm(**kwargs)
    torch.manual_seed(42);m=CachedModel().eval()
    a,ad=f.option_calibrated_flat_score(m,VariedProcessor(),prompts(),prompts(True),prefix_cache=True)
    assert m.cached_calls>0 and m.encodes==1
    b,bd=f.option_calibrated_flat_score(m,VariedProcessor(),prompts(),prompts(True),prefix_cache=False)
    assert a==pytest.approx(b,abs=2e-6)
    assert [d['calibrated_margin'] for d in ad]==pytest.approx([d['calibrated_margin'] for d in bd],abs=2e-6)
