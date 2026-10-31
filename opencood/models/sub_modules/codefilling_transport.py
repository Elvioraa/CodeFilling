"""Native UMGM and Information Filling with auditable logical wire accounting.

The upstream forward uses stochastic hard Gumbel samples even in eval; its returned
argmax indices need not describe those samples. Capture the *actual* decoder input
instead. No replacement quantizer or information-selection algorithm is used.
"""
import math
import torch
from torch import nn
from opencood.models.sub_modules.codebook import UMGMQuantizer
from opencood.models.comm_modules.where2comm import Communication


def make_quantizer(cfg):
    return UMGMQuantizer(64, cfg['seg_num'], [cfg['dict_size']] * cfg['r'], 0.0,
                         {key: lambda: nn.Linear(64, 64) for key in UMGMQuantizer._components})


def center_grid(x, height, width):
    """Origin-centered physical crop/pad at 0.8m. Never resize physical space."""
    h, w = x.shape[-2:]
    # Same integer center convention as torchvision CenterCrop.
    if h < height or w < width:
        ph, pw = max(height-h, 0), max(width-w, 0)
        x = torch.nn.functional.pad(x, (pw//2, (pw+1)//2, ph//2, (ph+1)//2))
    h, w = x.shape[-2:]
    y, z = int(round((h-height)/2.)), int(round((w-width)/2.))
    return x[..., y:y+height, z:z+width]


def communication_ledger(masks, record_len, cfg, channels=64, scalar_bits=32, coarse_channels=()):
    """Packed per-message index + bitmap representation, excluding local ego.

    Any nonzero fractional warped mask needs an index. Fractional amplitudes are
    additionally encoded as fp32. Dense confidence maps and pose metadata are
    reported separately; totals are logical protocol bytes, not measured traffic.
    """
    if masks.ndim != 4 or masks.shape[1] != 1 or sum(map(int, record_len)) != len(masks):
        raise ValueError('mask/record_len mismatch')
    h, w = masks.shape[-2:]
    bits_per_cell = cfg['seg_num'] * cfg['r'] * math.ceil(math.log2(cfg['dict_size']))
    rows, offset = [], 0
    for b, count in enumerate(map(int, record_len)):
        for a in range(1, count):
            mask = masks[offset+a].detach()
            selected = int(torch.count_nonzero(mask).item())
            fractional = int(torch.count_nonzero((mask != 0) & (mask != 1)).item())
            payload = math.ceil(selected * bits_per_cell / 8)
            bitmap = math.ceil(h*w / 8)
            weights = fractional * 4
            # The native graph computes coarse features BEFORE masking. Selected
            # fine indices alone cannot reconstruct those receptive fields. Count
            # the actual selected coarse vectors too, rather than underreporting.
            coarse_mask, coarse = mask.unsqueeze(0), []
            for channel_count in coarse_channels:
                coarse_mask = torch.nn.functional.max_pool2d(coarse_mask, 2)
                cells = int(torch.count_nonzero(coarse_mask).item())
                coarse.append(dict(channels=channel_count, selected_cells=cells,
                                   bytes=cells*channel_count*scalar_bits//8))
            coarse_bytes = sum(level['bytes'] for level in coarse)
            rows.append(dict(batch=b, sender=a, selected_cells=selected, fractional_cells=fractional,
                             index_bits=selected * bits_per_cell, index_bytes=payload, bitmap_bytes=bitmap,
                             mask_weight_bytes=weights, coarse_levels=coarse, coarse_bytes=coarse_bytes,
                             feature_bytes=payload+bitmap+weights+coarse_bytes,
                             confidence_bytes=h*w*4, pose_bytes=6*4,
                             dense_feature_bytes=h*w*channels*scalar_bits//8))
        offset += count
    return dict(kind='logical_packed_protocol_v1', measured_network_bytes=None,
                codebook_distribution='preinstalled; one-time checkpoint bytes excluded',
                control_policy='dense fp32 confidence per peer + 6 fp32 pose to ego; bitmap and fractional weights returned',
                multiscale_policy='finest sampled code indices + selected fp32 coarse vectors (native computes coarse before masking)',
                messages=rows, peers=len(rows), selected_cells=sum(r['selected_cells'] for r in rows),
                feature_bytes=sum(r['feature_bytes'] for r in rows),
                control_bytes=sum(r['confidence_bytes']+r['pose_bytes'] for r in rows),
                total_bytes=sum(r['feature_bytes']+r['confidence_bytes']+r['pose_bytes'] for r in rows))


class CodeFillingTransport(nn.Module):
    def __init__(self, codebook, communication):
        super().__init__()
        self.config = codebook
        self.quantizer = make_quantizer(codebook)
        self.communication = Communication(communication)
        self.last_codes = None

    def reconstruct(self, x, record_len):
        n, c, h, w = x.shape
        if c != 64 or sum(map(int, record_len)) != n:
            raise ValueError('64-channel packed feature contract violated')
        actual, hooks = {}, []
        for i, decoder in enumerate(self.quantizer._decoders):
            def capture(module, args, i=i):
                actual[i] = args[0].detach().argmax(-1)
            hooks.append(decoder._dequantizer.register_forward_pre_hook(capture))
        try:
            restored, _, _, loss = self.quantizer(x.permute(0, 2, 3, 1).reshape(-1, c))
        finally:
            for hook in hooks:
                hook.remove()
        self.last_codes = [actual[i].reshape(n, h, w, -1) for i in range(len(actual))]
        restored = restored.reshape(n, h, w, c).permute(0, 3, 1, 2).contiguous()
        # Preserve native local-ego bypass, without autograd in-place writes.
        ego = torch.zeros(n, dtype=torch.bool, device=x.device)
        offset = 0
        for count in map(int, record_len):
            ego[offset] = True
            offset += count
        restored = torch.where(ego[:, None, None, None], x, restored)
        return restored, loss

    def forward(self, x, confidence, record_len, transforms, enabled=True, coarse_channels=()):
        if enabled:
            x, loss = self.reconstruct(x, record_len)
        else:
            loss = x.sum() * 0
        _, masks, rate = self.communication(confidence, record_len, transforms)
        ledger = communication_ledger(masks, record_len, self.config, coarse_channels=coarse_channels)
        return x, masks, rate, loss, ledger


def pack_indices(indices, dictionary_size):
    """Reference fixed-width packing, MSB first, per-message byte padding."""
    width = math.ceil(math.log2(dictionary_size))
    accumulator, bits, output = 0, 0, bytearray()
    for value in indices:
        value = int(value)
        if not 0 <= value < dictionary_size:
            raise ValueError('code index outside dictionary')
        accumulator = (accumulator << width) | value
        bits += width
        while bits >= 8:
            bits -= 8
            output.append((accumulator >> bits) & 255)
            accumulator &= (1 << bits)-1
    if bits:
        output.append(accumulator << (8-bits))
    return bytes(output)


def unpack_indices(payload, count, dictionary_size):
    width = math.ceil(math.log2(dictionary_size))
    if len(payload) != math.ceil(count*width/8):
        raise ValueError('packed size mismatch')
    value = int.from_bytes(payload, 'big')
    padding = len(payload)*8-count*width
    if padding and value & ((1 << padding)-1):
        raise ValueError('nonzero padding')
    result = [(value >> (len(payload)*8-(i+1)*width)) & ((1 << width)-1) for i in range(count)]
    if any(x >= dictionary_size for x in result):
        raise ValueError('invalid packed index')
    return result


def decode_sampled_codes(quantizer, codes):
    """Exact native decoder on captured samples; avoids upstream unused _ix API."""
    result = None
    for decoder, code, size in zip(reversed(quantizer._decoders), reversed(codes), reversed(quantizer._k)):
        one_hot = torch.nn.functional.one_hot(code.reshape(-1, quantizer._m), size).float()
        result = decoder(one_hot, result)
    return result
