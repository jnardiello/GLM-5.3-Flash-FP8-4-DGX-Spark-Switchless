# Host controls

`tp4-iommu.sh` manages the host IOMMU setting outside the container.
`scripts/deploy-host.sh` copies it to `~/tp4/host/` with SHA-256 and shell-syntax
verification. It never reboots a node.

```sh
./scripts/deploy-host.sh
./scripts/deploy-host.sh --run tp4-iommu.sh --apply
./scripts/deploy-host.sh --no-push --run tp4-iommu.sh --status
```

| Script | Effect | Revert | Production state |
| --- | --- | --- | --- |
| `tp4-iommu.sh` | verifies and regenerates GRUB so `iommu.passthrough=1` wins for the next boot | `--revert`, then rolling reboot | applied on all four verified nodes |

The script uses `--apply`, `--revert`, or `--status`, is idempotent, and verifies the
read-back value. Exit codes are `0` success, `2` usage, `3` unsupported, `4` unsafe
revert/unknown state, and `5` missing passwordless sudo. Exit code 4 from an IOMMU
revert means GRUB was not safely regenerated: do not reboot.

The IOMMU setting is a boot-time flag. Apply or revert it only in an authorized host
window, stop TP4 first, then reboot rank 3→2→1→0 with rank 0 last. Afterward require
`--status` on all ranks and a green `./scripts/tp4ctl fabric-check`. The revert sentinel under
`/etc/default/grub.d/` prevents a later routine push from resurrecting a deliberately
removed drop-in.

See [`docs/operations.md`](../../../docs/operations.md) for authorization and recovery,
and [`docs/production-recipe.md`](../../../docs/production-recipe.md) for the current host
tier.
