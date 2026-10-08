"""
Extract PNG frames from EgoClip clip mp4s.

STP pre-training reads individual frames by absolute index (frame t and t+16),
so it is much faster to decode each clip once up front than to seek into mp4s
every step. Builds:

    {output_root}/{video_uid}/{narration_source}_{idx}_clip/{frame_idx}.png

Input layout expected:  {clip_root}/{video_uid}/{narration_source}_{idx}_clip.mp4

Usage:
    python tools/prepare_ego4d_frames.py \
        --manifest /path/to/egoclip.csv \
        --clip_root /data/ego4d_256_egoclip_AVG3s \
        --output_root /data/ego4d_256_egoclip_AVG3s_img \
        --workers 16 --shard 0 --num_shards 4
"""
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from decord import VideoReader
from PIL import Image


def process_clip(video_path, output_dir, overwrite=False):
    output_dir = Path(output_dir)
    if output_dir.exists() and not overwrite and any(output_dir.iterdir()):
        return 'skipped'

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        vr = VideoReader(str(video_path))
        for frame_idx, frame in enumerate(vr):
            Image.fromarray(frame.asnumpy()).save(str(output_dir / f'{frame_idx}.png'))
        vr.close()
        return 'ok'
    except Exception as e:                                   # noqa: BLE001
        return f'error: {type(e).__name__}: {e}'


def main():
    p = argparse.ArgumentParser('EgoClip mp4 -> PNG frames')
    p.add_argument('--manifest', required=True)
    p.add_argument('--clip_root', required=True,
                   help='root containing {video_uid}/{clip_name}.mp4')
    p.add_argument('--output_root', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num_shards', type=int, default=1,
                   help='split the manifest across N processes/nodes')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--limit', type=int, default=0, help='only do the first N clips')
    args = p.parse_args()

    with open(args.manifest, newline='') as fp:
        manifest = list(csv.DictReader(fp))
    if args.limit:
        manifest = manifest[:args.limit]
    manifest = manifest[args.shard::args.num_shards]
    print(f'shard {args.shard}/{args.num_shards}: {len(manifest)} clips')

    jobs = []
    for clip in manifest:
        clip_name = '{}_{}_clip'.format(clip['narration_source'], clip['narration_ind'])
        src = os.path.join(args.clip_root, clip['video_uid'], clip_name + '.mp4')
        dst = os.path.join(args.output_root, clip['video_uid'], clip_name)
        jobs.append((src, dst, args.overwrite))

    stats = {'ok': 0, 'skipped': 0, 'missing': 0, 'error': 0}
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for src, dst, ow in jobs:
            if not os.path.exists(src):
                stats['missing'] += 1
                continue
            result = ex.submit(process_clip, src, dst, ow)
            status = result.result()
            key = status if status in stats else 'error'
            stats[key] += 1
            if key == 'error':
                errors.append((dst, status))

    print('done:', stats)
    for path, err in errors[:10]:
        print('  ', path, err)


if __name__ == '__main__':
    main()
