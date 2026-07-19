import torch


def buggy_cdf_sample_from_log_probs(
    log_probs: torch.Tensor,
    mask: torch.Tensor,
    threshold: torch.Tensor,
) -> torch.Tensor:
    assert log_probs.ndim == 2, tuple(log_probs.shape)
    assert mask.shape == log_probs.shape, (tuple(mask.shape), tuple(log_probs.shape))
    assert mask.dtype == torch.bool, mask.dtype
    assert threshold.shape == (log_probs.shape[0], 1), tuple(threshold.shape)

    probs = torch.where(
        mask,
        log_probs.to(dtype=torch.float32).exp(),
        torch.zeros_like(log_probs, dtype=torch.float32),
    )
    cdf = probs.cumsum(dim=-1)
    cdf_crossed = cdf > threshold
    return cdf_crossed.to(dtype=torch.int64).argmax(dim=-1).to(dtype=torch.long)


def fixed_cdf_sample_from_log_probs(
    log_probs: torch.Tensor,
    mask: torch.Tensor,
    threshold: torch.Tensor,
) -> torch.Tensor:
    assert log_probs.ndim == 2, tuple(log_probs.shape)
    assert mask.shape == log_probs.shape, (tuple(mask.shape), tuple(log_probs.shape))
    assert mask.dtype == torch.bool, mask.dtype
    assert threshold.shape == (log_probs.shape[0], 1), tuple(threshold.shape)

    probs = torch.where(
        mask,
        log_probs.to(dtype=torch.float32).exp(),
        torch.zeros_like(log_probs, dtype=torch.float32),
    )
    cdf = probs.cumsum(dim=-1)
    total = cdf[..., -1:]
    zero = torch.zeros_like(total)
    threshold_below_total = torch.minimum(threshold, torch.nextafter(total, zero))
    cdf_crossed = cdf > threshold_below_total
    return cdf_crossed.to(dtype=torch.int64).argmax(dim=-1).to(dtype=torch.long)


def main() -> None:
    dst_mask = torch.tensor([[False, False, True, False]], dtype=torch.bool)
    dst_log_probs = torch.full((1, 4), float("-inf"), dtype=torch.float32)
    dst_log_probs[dst_mask] = 0.0

    dst_probs = torch.where(
        dst_mask,
        dst_log_probs.exp(),
        torch.zeros_like(dst_log_probs),
    )
    threshold_at_total = dst_probs.cumsum(dim=-1)[..., -1:]

    buggy_dst = buggy_cdf_sample_from_log_probs(
        dst_log_probs,
        dst_mask,
        threshold_at_total,
    )
    fixed_dst = fixed_cdf_sample_from_log_probs(
        dst_log_probs,
        dst_mask,
        threshold_at_total,
    )

    amount_mask = torch.tensor(
        [[[False, False, False], [False, False, False], [False, True, False], [False, False, False]]],
        dtype=torch.bool,
    )
    buggy_amount_mask_at_dst = torch.gather(
        amount_mask,
        1,
        buggy_dst.view(1, 1, 1).expand(1, 1, amount_mask.shape[-1]),
    ).squeeze(1)
    fixed_amount_mask_at_dst = torch.gather(
        amount_mask,
        1,
        fixed_dst.view(1, 1, 1).expand(1, 1, amount_mask.shape[-1]),
    ).squeeze(1)

    print(f"dst_mask={dst_mask.tolist()}")
    print(f"threshold_at_total={threshold_at_total.flatten().tolist()}")
    print(f"buggy_dst={buggy_dst.tolist()}")
    print(f"buggy_amount_mask_at_dst={buggy_amount_mask_at_dst.tolist()}")
    print(f"fixed_dst={fixed_dst.tolist()}")
    print(f"fixed_amount_mask_at_dst={fixed_amount_mask_at_dst.tolist()}")

    assert int(buggy_dst.item()) == 0
    assert not bool(buggy_amount_mask_at_dst.any(dim=-1).item())
    assert int(fixed_dst.item()) == 2
    assert bool(fixed_amount_mask_at_dst.any(dim=-1).item())


if __name__ == "__main__":
    main()
