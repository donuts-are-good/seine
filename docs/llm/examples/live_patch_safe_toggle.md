# Example: Live Patch Request

```bash
curl -s -X PATCH http://127.0.0.1:9977/v1/miner/live-config \
  -H 'content-type: application/json' \
  -d '{"stats_secs":3,"work_allocation":"adaptive"}'
```

If `requires_restart=true`, apply a restart flow.
