"""Transcribe candidate tracks locally without priming recognition with the target text."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--model', default='base')
    parser.add_argument('--language', default='zh')
    parser.add_argument('--files', nargs='+', default=['speaker_1.wav', 'speaker_2.wav'])
    args = parser.parse_args()
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device='cpu', compute_type='int8',
                         cpu_threads=4, download_root=str(args.cache))
    for name in args.files:
        path = args.folder / name
        segments, _ = model.transcribe(str(path), language=args.language, beam_size=5,
                                       vad_filter=True, condition_on_previous_text=False)
        rows = [{'start':s.start, 'end':s.end, 'text':s.text} for s in segments]
        path.with_name(path.stem + '_transcript.json').write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'file':name, 'segments':rows}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
