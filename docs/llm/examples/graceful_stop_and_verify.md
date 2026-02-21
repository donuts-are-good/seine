# Example: Graceful Stop and Verify

```bash
curl -s -X POST http://127.0.0.1:9977/v1/miner/stop -H 'content-type: application/json' -d '{"wait_ms":15000}'
curl -s http://127.0.0.1:9977/v1/runtime/state
```
