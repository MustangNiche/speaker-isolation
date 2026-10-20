"""Local two-speaker separation and explicit, verified track export."""
import argparse
import json
import pathlib
import subprocess
import sys
import time
from types import SimpleNamespace


def fresh_directory(path):
    path = pathlib.Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Choose an empty output directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_audio(path, rate):
    import av
    import numpy as np
    resampler = av.AudioResampler(format='fltp', layout='mono', rate=rate)
    chunks = []
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            chunks.extend(f.to_ndarray()[0] for f in resampler.resample(frame))
    chunks.extend(f.to_ndarray()[0] for f in resampler.resample(None))
    if not chunks:
        raise ValueError('The input has no decodable audio')
    audio = np.concatenate(chunks)
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-8:
        raise ValueError('Audio is silent or contains invalid samples')
    return audio


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def separate(args):
    import numpy as np
    import torch
    import soundfile as sf
    if not args.reference_text.strip():
        raise ValueError('First obtain a sentence spoken by the person to keep')
    assets = pathlib.Path(args.assets).resolve()
    source = pathlib.Path(args.input).resolve(strict=True)
    output = fresh_directory(args.output)
    sys.path.insert(0, str(assets / 'MossFormer2'))
    from mossformer2_ss.mossformer2 import MossFormer2_SS_16K
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    model = MossFormer2_SS_16K(SimpleNamespace(
        encoder_embedding_dim=512, mossformer_sequence_dim=512,
        num_mossformer_layer=24, encoder_kernel_size=16, num_spks=2))
    checkpoint = torch.load(assets / 'MossFormer2/last_best_checkpoint.pt',
                            map_location='cpu', weights_only=True)
    for key in ('model', 'state_dict', 'model_state_dict'):
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break
    checkpoint = {k.removeprefix('module.'): v for k, v in checkpoint.items()}
    if not next(iter(checkpoint)).startswith('model.'):
        checkpoint = {'model.' + k: v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    del checkpoint
    audio = read_audio(source, 16000)
    if args.duration is not None:
        if args.duration <= 0:
            raise ValueError('duration must be positive')
        audio = audio[:round(args.duration * 16000)]
    sf.write(output / 'original_16k.wav', audio, 16000, subtype='FLOAT')
    window, step = 96000, 48000
    total = np.zeros((2, len(audio) + window), dtype=np.float32)
    weights = np.zeros(len(audio) + window, dtype=np.float32)
    fade = np.minimum(np.arange(window) / step,
                      (window - np.arange(window)) / step).clip(0.001, 1).astype(np.float32)
    starts = [0]
    # Stop once a full window covers the end; avoid ambiguous tiny final chunks.
    while starts[-1] + window < len(audio):
        starts.append(starts[-1] + step)
    log = []
    with torch.inference_mode():
        for start in starts:
            began = time.monotonic()
            clip = audio[start:start + window]
            valid = len(clip)
            clip = np.pad(clip, (0, window - valid))
            pred = np.stack([t[0].numpy() for t in model(torch.from_numpy(clip)[None])])
            confidence = None
            if start == 0:
                permutation = [0, 1] if np.sum(pred[0] ** 2) >= np.sum(pred[1] ** 2) else [1, 0]
            else:
                overlap = min(step, valid)
                past = total[:, start:start + overlap] / np.maximum(weights[start:start + overlap], 1e-8)
                current = pred[:, :overlap]
                corr = (past @ current.T) / np.maximum(
                    np.linalg.norm(past, axis=1)[:, None] * np.linalg.norm(current, axis=1)[None], 1e-8)
                score = float(corr[0, 0] + corr[1, 1] - corr[0, 1] - corr[1, 0])
                confidence = abs(score)
                permutation = [0, 1] if score >= 0 else [1, 0]
            total[:, start:start + window] += pred[permutation] * fade
            weights[start:start + window] += fade
            item = {'start_seconds': start / 16000, 'permutation': permutation,
                    'identity_confidence': confidence,
                    'needs_identity_review': confidence is not None and confidence < 0.25,
                    'processing_seconds': round(time.monotonic() - began, 2)}
            log.append(item)
            print(json.dumps(item), flush=True)
    separated = total[:, :len(audio)] / np.maximum(weights[:len(audio)], 1e-8)
    if not np.isfinite(separated).all():
        raise ValueError('Invalid model output')
    for index in range(2):
        sf.write(output / f'speaker_{index + 1}.wav', separated[index], 16000, subtype='FLOAT')
    write_json(output / 'separation.json', {
        'source': str(source), 'reference_text': args.reference_text,
        'reference_is_identity_check_not_model_conditioning': True,
        'model': 'MossFormer2_SS_16K', 'sample_rate': 16000,
        'samples': len(audio), 'duration': len(audio) / 16000,
        'preview_only': args.duration is not None, 'chunks': log})
    print('Choose a track by checking the target sentence; neither track is automatically selected.', flush=True)


def export(args):
    import av
    import numpy as np
    import soundfile as sf
    if pathlib.Path(args.name).name != args.name or any(c in args.name for c in '/\\:'):
        raise ValueError('name must be a plain filename stem')
    separation = pathlib.Path(args.separation).resolve()
    metadata = json.loads((separation / 'separation.json').read_text(encoding='utf-8'))
    if metadata['preview_only'] and not args.allow_preview:
        raise ValueError('This is a preview; run full separation or explicitly pass --allow-preview')
    binary = pathlib.Path(args.assets).resolve() / 'DeepFilterNet' / (
        'deep-filter.exe' if sys.platform == 'win32' else 'deep-filter')
    if not binary.is_file():
        raise FileNotFoundError(binary)
    output = fresh_directory(args.output)
    work = output / 'processing'
    work.mkdir()
    audio = read_audio(separation / f'speaker_{args.track}.wav', 48000)
    scale = min(2., 0.8 / max(float(np.max(np.abs(audio))), 1e-8))
    source = work / 'isolated.wav'
    sf.write(source, np.pad(audio * scale, (0, 4800)), 48000, subtype='PCM_16')
    cleaned = work / 'clean'
    cleaned.mkdir()
    result = subprocess.run([str(binary), '-D', '--pf', '--pf-beta', '0.03',
                             '-o', str(cleaned), str(source)],
                            capture_output=True, text=True, errors='replace')
    (work / 'deepfilter.log').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    result.check_returncode()
    clean = read_audio(cleaned / source.name, 48000)
    if not len(audio) <= len(clean) <= len(audio) + 4800:
        raise ValueError('Unexpected denoiser output length')
    clean = clean[:len(audio)]
    clean *= min(2., 0.9 / max(float(np.max(np.abs(clean))), 1e-8))
    reports = []
    for suffix, codec in (('.wav', 'pcm_s24le'), ('.mp3', 'libmp3lame')):
        path = output / (args.name + suffix)
        with av.open(str(path), 'w') as container:
            stream = container.add_stream(codec, rate=48000)
            stream.layout = 'mono'
            if suffix == '.mp3':
                stream.bit_rate = 192000
            for start in range(0, len(clean), 4096):
                frame = av.AudioFrame.from_ndarray(
                    np.ascontiguousarray(clean[None, start:start + 4096], dtype=np.float32),
                    format='fltp', layout='mono')
                frame.sample_rate = 48000
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
        verify = read_audio(path, 48000)
        peak = float(np.max(np.abs(verify)))
        if abs(len(verify) - len(clean)) > 10 or peak >= 1:
            raise ValueError(f'Export validation failed: {path}')
        reports.append({'file': path.name, 'duration': len(verify) / 48000, 'peak': peak})
    sf.write(output / 'narrator_final_16k.wav',
             read_audio(output / (args.name + '.wav'), 16000), 16000, subtype='PCM_16')
    write_json(output / 'export.json', {'selected_track': args.track,
               'reference_text': metadata['reference_text'], 'original_mixed_back': False,
               'preview_only': metadata['preview_only'], 'files': reports})
    print(json.dumps(reports, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    sep = commands.add_parser('separate')
    sep.add_argument('--input', required=True)
    sep.add_argument('--output', required=True)
    sep.add_argument('--assets', required=True)
    sep.add_argument('--reference-text', required=True)
    sep.add_argument('--threads', type=int, default=6)
    sep.add_argument('--duration', type=float)
    sep.set_defaults(run=separate)
    exp = commands.add_parser('export')
    exp.add_argument('--separation', required=True)
    exp.add_argument('--track', type=int, choices=[1, 2], required=True)
    exp.add_argument('--assets', required=True)
    exp.add_argument('--output', required=True)
    exp.add_argument('--name', default='主播提取版')
    exp.add_argument('--allow-preview', action='store_true')
    exp.set_defaults(run=export)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
