"""
Upload the built release folder to HuggingFace.

    # 1) log in once, in your own terminal (the token is stored under
    #    ~/.cache/huggingface/token and never needs to be pasted anywhere else)
    hf auth login

    # 2) build the slim files
    python tools/make_hf_release.py --ckpt weights/lang_0.95_aug0.8_no_lang.pth

    # 3) preview, then upload
    python tools/upload_hf.py --repo-id MCG-NJU/STP --dry-run
    python tools/upload_hf.py --repo-id MCG-NJU/STP

This script reads the token from the standard HuggingFace cache via
`huggingface_hub`, so no credential is ever passed on the command line or
printed. If no token is found it stops and tells you to run `hf auth login`.
"""
import argparse
import os
import sys


def check_credentials():
    """Return the logged-in user, or exit with instructions."""
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import LocalTokenNotFoundError
    except ImportError:
        sys.exit('huggingface_hub is not installed. Run: pip install huggingface_hub')

    try:
        return HfApi().whoami()['name']
    except LocalTokenNotFoundError:
        sys.exit(
            'No HuggingFace token found.\n'
            'Run this in your own terminal (not through a script):\n\n'
            '    hf auth login\n\n'
            'It will prompt for a token from https://huggingface.co/settings/tokens\n'
            'with *write* permission, and store it in ~/.cache/huggingface/token.\n'
            'Then re-run this script.')


def main():
    p = argparse.ArgumentParser('upload the STP release to HuggingFace')
    p.add_argument('--repo-id', required=True, help='e.g. MCG-NJU/STP')
    p.add_argument('--folder', default='hf_release')
    p.add_argument('--private', action='store_true')
    p.add_argument('--dry-run', action='store_true',
                   help='list what would be uploaded, and verify auth only')
    p.add_argument('--commit-message', default='Add STP ViT-B/16 pre-trained weights')
    args = p.parse_args()

    user = check_credentials()
    print(f'[hf] authenticated as: {user}')

    from huggingface_hub import HfApi
    api = HfApi()

    if not os.path.isdir(args.folder):
        sys.exit(f'folder not found: {args.folder}\n'
                 f'Build it first: python tools/make_hf_release.py --ckpt <path>')

    files = sorted(f for f in os.listdir(args.folder)
                   if os.path.isfile(os.path.join(args.folder, f)))
    total = sum(os.path.getsize(os.path.join(args.folder, f)) for f in files)
    print(f'[hf] {len(files)} files, {total/1048576:.1f} MiB total:')
    for f in files:
        print(f'       {os.path.getsize(os.path.join(args.folder, f))/1048576:8.1f} MiB  {f}')

    owner = args.repo_id.split('/')[0]
    if owner not in (user,) and '/' in args.repo_id:
        print(f'[hf] note: uploading to the `{owner}` namespace, '
              f'which requires write access there.')

    if args.dry_run:
        print('[hf] dry run: authenticated and files present, nothing uploaded.')
        return

    api.create_repo(repo_id=args.repo_id, repo_type='model',
                    private=args.private, exist_ok=True)
    print(f'[hf] repo ready: {args.repo_id}')

    api.upload_folder(
        repo_id=args.repo_id,
        repo_type='model',
        folder_path=args.folder,
        commit_message=args.commit_message,
    )
    print(f'[hf] done -> https://huggingface.co/{args.repo_id}')


if __name__ == '__main__':
    main()
