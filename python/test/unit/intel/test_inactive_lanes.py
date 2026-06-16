import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu


@triton.jit
def _inactive_lane_kernel(query_start_len, values, out, row_sums,
                          BLOCK: tl.constexpr):
    seq_idx = tl.program_id(0)
    start = tl.load(query_start_len + seq_idx)
    stop = tl.load(query_start_len + seq_idx + 1)
    row_len = stop - start

    if row_len == 0:
        return

    offsets = tl.arange(0, BLOCK)
    mask = offsets < row_len
    row_values = tl.load(values + seq_idx * BLOCK + offsets,
                         mask=mask,
                         other=-32768.0)

    best = tl.max(row_values, axis=0)
    total = tl.sum(tl.where(mask, row_values, 0.0), axis=0)

    tl.store(out + seq_idx, best)
    tl.store(row_sums + seq_idx, total)


@pytest.fixture(params=[None, False, True],
                ids=["default-io", "branch-io", "predicated-io"])
def predicated_io(request, monkeypatch, fresh_triton_cache):
    if request.param is None:
        monkeypatch.delenv("TRITON_INTEL_PREDICATED_LOAD", raising=False)
        monkeypatch.delenv("TRITON_INTEL_PREDICATED_STORE", raising=False)
    elif request.param is False:
        monkeypatch.setenv("TRITON_INTEL_PREDICATED_LOAD", "0")
        monkeypatch.setenv("TRITON_INTEL_PREDICATED_STORE", "0")
    elif request.param is True:
        monkeypatch.setenv("TRITON_INTEL_PREDICATED_LOAD", "1")
        monkeypatch.setenv("TRITON_INTEL_PREDICATED_STORE", "1")
    yield


@pytest.mark.skipif(not is_xpu(), reason="XPU-specific inactive-lane regression")
@pytest.mark.parametrize("batch", [2, 12])
def test_one_live_row_in_static_batch(batch, predicated_io, device):
    block = 16
    query_start_len = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    query_start_len[1:] = 1

    values = torch.full((batch, block),
                        -128.0,
                        dtype=torch.float32,
                        device=device)
    values[0, 0] = 42.0
    values[0, 1] = -7.0

    out = torch.full((batch, ), -999.0, dtype=torch.float32, device=device)
    row_sums = torch.full((batch, ), -999.0, dtype=torch.float32, device=device)

    _inactive_lane_kernel[(batch, )](
        query_start_len,
        values,
        out,
        row_sums,
        BLOCK=block,
        num_warps=1,
    )

    torch.testing.assert_close(out[0], torch.tensor(42.0, device=device))
    torch.testing.assert_close(row_sums[0], torch.tensor(42.0, device=device))
    torch.testing.assert_close(out[1:],
                               torch.full((batch - 1, ), -999.0,
                                          device=device))
    torch.testing.assert_close(row_sums[1:],
                               torch.full((batch - 1, ), -999.0,
                                          device=device))
