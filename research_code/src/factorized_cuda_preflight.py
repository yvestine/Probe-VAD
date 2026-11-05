"""NVIDIA runtime check for the repository's pinned Linux/CUDA environment."""
import argparse
import importlib.metadata
import json
import platform
import shutil
import sys

PINNED = {
    'torch': '2.5.1', 'torchvision': '0.20.1', 'transformers': '4.46.3',
    'accelerate': '0.26.0', 'numpy': '1.25.2', 'scipy': '1.9.3',
    'scikit-learn': '1.1.3', 'opencv-python': '4.11.0.86',
    'pillow': '10.4.0', 'huggingface-hub': '0.30.2',
    'av': '14.2.0', 'decord': '0.6.0', 'qwen-vl-utils': '0.0.10', 'tqdm': '4.67.1',
}

def version_errors(versions, python_version):
    errors=[]
    if tuple(python_version[:2]) != (3,10):
        errors.append('Use Python 3.10 as specified in environment.yml')
    for name, expected in PINNED.items():
        actual=versions.get(name, 'missing')
        if actual.split('+')[0] != expected:
            errors.append(f'{name}: expected {expected}, found {actual}')
    return errors

def check(attention='flash_attention_2', device='cuda:0'):
    versions={}
    for name in (*PINNED, 'flash-attn', 'ffmpeg-python', 'imageio', 'einops'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    errors=version_errors(versions,sys.version_info)
    if platform.system()!='Linux': errors.append('This production launcher targets Linux/NVIDIA')
    if not device.startswith('cuda:'): errors.append('CUDA device required; no CPU/MPS fallback')
    for executable in ('ffmpeg','ffprobe'):
        if not shutil.which(executable): errors.append(f'Missing executable: {executable}')
    if errors: raise RuntimeError('\n'.join(errors))
    # Remote VideoLLaMA3 processor imports these beyond upstream requirements.txt.
    import ffmpeg, imageio, einops
    if not callable(getattr(ffmpeg,'input',None)):
        raise RuntimeError('Install ffmpeg-python, not the unrelated ffmpeg package')
    import torch
    if torch.version.cuda != '12.4':
        raise RuntimeError(f'Expected CUDA 12.4 PyTorch build; found {torch.version.cuda}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: check NVIDIA driver and CUDA_VISIBLE_DEVICES')
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('Original loader uses BF16; this GPU must support BF16')
        # Catch driver/build/architecture incompatibility, not merely device discovery.
        x=torch.ones((32,32),device=device,dtype=torch.bfloat16)
        value=(x@x).float().mean().item()
        if value != 32: raise RuntimeError('CUDA BF16 matrix multiplication failed')
        if attention=='flash_attention_2':
            from flash_attn import flash_attn_func
            q=torch.zeros((1,4,2,32),device=device,dtype=torch.bfloat16)
            result=flash_attn_func(q,q,q,causal=True)
            torch.cuda.synchronize()
            if not torch.isfinite(result).all().item(): raise RuntimeError('FlashAttention kernel failed')
        elif attention!='eager': raise ValueError('Use flash_attention_2 or eager')
        report=dict(python=platform.python_version(),packages=versions,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(device),capability=list(torch.cuda.get_device_capability(device)),
            attention=attention,device=device,bf16_kernel_passed=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attn_implementation',choices=['flash_attention_2','eager'],default='flash_attention_2')
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    try: print(json.dumps(check(args.attn_implementation,args.device),indent=2))
    except Exception as exc: p.exit(2,f'NVIDIA preflight failed:\n{exc}\n')
if __name__=='__main__': main()
