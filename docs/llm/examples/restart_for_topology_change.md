# Example: Restart for Session Changes

```bash
curl -s -X POST http://127.0.0.1:9977/v1/miner/stop
curl -s -X POST http://127.0.0.1:9977/v1/miner/start -H 'content-type: application/json' -d '{"threads":2}'
```
