"""
Standalone verification that our clean `models_stp.py` reproduces the released
checkpoint's exact parameter set -- with NO timm / torch installed.

It parses the parameter names + shapes straight out of the torch-1.13 zip-format
checkpoint via a torch stub, rebuilds the STP parameter shapes from first
principles, and diffs the two sets.

    python tools/verify_ckpt_params.py lang_0.95_aug0.8_no_lang.pth

Expected output: "434/434 tensors matched, 0 missing, 0 unexpected".
"""
import io
import pickle
import sys
import types
import zipfile


# ---------------------------------------------------------------------------
# minimal torch stub so the pickle can be unpickled on a machine without torch
# ---------------------------------------------------------------------------
class _Stub:
    def __init__(self, *a, **k):
        pass


def _shape_holder(size):
    class _T:
        shape = tuple(size)
    return _T()


def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad, hook=None, **kw):
    return _shape_holder(size)


def _rebuild_tensor(storage, offset, size, stride):
    return _shape_holder(size)


def _rebuild_parameter(data, requires_grad, hook=None):
    return data


def _install_torch_stub():
    torch = types.ModuleType('torch')
    torch.__path__ = []
    utils = types.ModuleType('torch._utils')
    utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    utils._rebuild_tensor = _rebuild_tensor
    utils._rebuild_parameter = _rebuild_parameter
    storage = types.ModuleType('torch.storage')
    storage._TypedStorage = _Stub
    torch.storage = storage
    torch._utils = utils
    torch.FloatStorage = _Stub
    sys.modules.update({'torch': torch, 'torch._utils': utils, 'torch.storage': storage})


def read_checkpoint_param_shapes(path):
    _install_torch_stub()
    with zipfile.ZipFile(path) as z:
        unpickler = pickle.Unpickler(io.BytesIO(z.read('archive/data.pkl')), encoding='latin-1')
        unpickler.persistent_load = lambda pid: _Stub()
        ckpt = unpickler.load()
    state = ckpt['model'] if 'model' in ckpt else ckpt
    return {k: tuple(v.shape) for k, v in state.items()}, ckpt


# ---------------------------------------------------------------------------
# STP parameter shape table, rebuilt from the architecture definition
# ---------------------------------------------------------------------------
def stp_param_shapes(embed_dim=768, depth=12, num_heads=12, patch=16, in_chans=3,
                     dec_dim=512, dec_depth=8, dec_heads=16, mlp_ratio=4,
                     img_size=224):
    L = (img_size // patch) ** 2
    s = {}

    # encoder
    s['cls_token'] = (1, 1, embed_dim)
    s['pos_embed'] = (1, L + 1, embed_dim)
    s['patch_embed.proj.weight'] = (embed_dim, in_chans, patch, patch)
    s['patch_embed.proj.bias'] = (embed_dim,)
    for i in range(depth):
        p = f'blocks.{i}.'
        s[p + 'norm1.weight'] = (embed_dim,)
        s[p + 'norm1.bias'] = (embed_dim,)
        s[p + 'attn.qkv.weight'] = (3 * embed_dim, embed_dim)
        s[p + 'attn.qkv.bias'] = (3 * embed_dim,)
        s[p + 'attn.proj.weight'] = (embed_dim, embed_dim)
        s[p + 'attn.proj.bias'] = (embed_dim,)
        s[p + 'norm2.weight'] = (embed_dim,)
        s[p + 'norm2.bias'] = (embed_dim,)
        s[p + 'mlp.fc1.weight'] = (embed_dim * mlp_ratio, embed_dim)
        s[p + 'mlp.fc1.bias'] = (embed_dim * mlp_ratio,)
        s[p + 'mlp.fc2.weight'] = (embed_dim, embed_dim * mlp_ratio)
        s[p + 'mlp.fc2.bias'] = (embed_dim,)
    s['norm.weight'] = (embed_dim,)
    s['norm.bias'] = (embed_dim,)

    # shared decoder bits
    s['mask_token'] = (1, 1, dec_dim)
    s['decoder_pos_embed'] = (1, L + 1, dec_dim)
    s['decoder_embed_1.weight'] = (dec_dim, embed_dim)
    s['decoder_embed_1.bias'] = (dec_dim,)
    s['decoder_embed_2.weight'] = (dec_dim, embed_dim)
    s['decoder_embed_2.bias'] = (dec_dim,)
    s['decoder_norm.weight'] = (dec_dim,)
    s['decoder_norm.bias'] = (dec_dim,)
    s['decoder_pred_1.weight'] = (patch * patch * in_chans, dec_dim)
    s['decoder_pred_1.bias'] = (patch * patch * in_chans,)
    s['decoder_pred_2.weight'] = (patch * patch * in_chans, dec_dim)
    s['decoder_pred_2.bias'] = (patch * patch * in_chans,)

    # spatial decoder: plain transformer blocks (timm Block naming)
    for i in range(dec_depth):
        p = f'decoder_blocks_1.{i}.'
        s[p + 'norm1.weight'] = (dec_dim,)
        s[p + 'norm1.bias'] = (dec_dim,)
        s[p + 'attn.qkv.weight'] = (3 * dec_dim, dec_dim)
        s[p + 'attn.qkv.bias'] = (3 * dec_dim,)
        s[p + 'attn.proj.weight'] = (dec_dim, dec_dim)
        s[p + 'attn.proj.bias'] = (dec_dim,)
        s[p + 'norm2.weight'] = (dec_dim,)
        s[p + 'norm2.bias'] = (dec_dim,)
        s[p + 'mlp.fc1.weight'] = (dec_dim * mlp_ratio, dec_dim)
        s[p + 'mlp.fc1.bias'] = (dec_dim * mlp_ratio,)
        s[p + 'mlp.fc2.weight'] = (dec_dim, dec_dim * mlp_ratio)
        s[p + 'mlp.fc2.bias'] = (dec_dim,)

    # temporal decoder: self-attn + cross-attn + FFN (DecoderBlock)
    for i in range(dec_depth):
        p = f'decoder_blocks_2.{i}.'
        s[p + 'norm1.weight'] = (dec_dim,)
        s[p + 'norm1.bias'] = (dec_dim,)
        s[p + 'self_attn.qkv.weight'] = (3 * dec_dim, dec_dim)
        s[p + 'self_attn.qkv.bias'] = (3 * dec_dim,)
        s[p + 'self_attn.proj.weight'] = (dec_dim, dec_dim)
        s[p + 'self_attn.proj.bias'] = (dec_dim,)
        s[p + 'cross_attn.q.weight'] = (dec_dim, dec_dim)
        s[p + 'cross_attn.q.bias'] = (dec_dim,)
        s[p + 'cross_attn.kv.weight'] = (2 * dec_dim, dec_dim)
        s[p + 'cross_attn.kv.bias'] = (2 * dec_dim,)
        s[p + 'cross_attn.proj.weight'] = (dec_dim, dec_dim)
        s[p + 'cross_attn.proj.bias'] = (dec_dim,)
        s[p + 'query_norm.weight'] = (dec_dim,)
        s[p + 'query_norm.bias'] = (dec_dim,)
        s[p + 'context_norm.weight'] = (dec_dim,)
        s[p + 'context_norm.bias'] = (dec_dim,)
        s[p + 'norm2.weight'] = (dec_dim,)
        s[p + 'norm2.bias'] = (dec_dim,)
        s[p + 'mlp.fc1.weight'] = (dec_dim * mlp_ratio, dec_dim)
        s[p + 'mlp.fc1.bias'] = (dec_dim * mlp_ratio,)
        s[p + 'mlp.fc2.weight'] = (dec_dim, dec_dim * mlp_ratio)
        s[p + 'mlp.fc2.bias'] = (dec_dim,)

    return s


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'lang_0.95_aug0.8_no_lang.pth'
    ckpt_shapes, ckpt = read_checkpoint_param_shapes(path)
    ours = stp_param_shapes()

    missing = sorted(set(ours) - set(ckpt_shapes))
    unexpected = sorted(set(ckpt_shapes) - set(ours))
    mismatched = sorted(k for k in set(ours) & set(ckpt_shapes) if ours[k] != ckpt_shapes[k])

    n_params = 0
    for k, v in ckpt_shapes.items():
        p = 1
        for d in v:
            p *= d
        n_params += p

    print(f'checkpoint      : {path}')
    print(f'checkpoint meta : {sorted(ckpt.keys())}')
    print(f'epoch           : {ckpt.get("epoch")}')
    print(f'tensors on disk : {len(ckpt_shapes)}')
    print(f'tensors expected: {len(ours)}')
    print(f'parameters      : {n_params/1e6:.2f} M')
    print()
    print(f'missing    ({len(missing)})  : {missing[:5]}')
    print(f'unexpected ({len(unexpected)}): {unexpected[:5]}')
    print(f'mismatched ({len(mismatched)}): {mismatched[:5]}')
    print()

    ok = not missing and not unexpected and not mismatched
    print('RESULT: %d/%d tensors matched' % (len(ours) - len(missing) - len(mismatched), len(ours)))
    print('RESULT:', 'OK -- strict=True load will succeed' if ok else 'MISMATCH')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
