# Example: Sparse Config Start (CPU-focused)

```bash
curl -s -X POST http://127.0.0.1:9977/v1/miner/start \
  -H 'content-type: application/json' \
  -d '{"threads":4,"cpu_profile":"throughput","work_allocation":"adaptive"}'
```
