# Example: Service Boot Idle -> Start

```bash
./seine --service --api-bind 127.0.0.1:9977 --token <daemon-token>
curl -s http://127.0.0.1:9977/v1/runtime/state
curl -s -X POST http://127.0.0.1:9977/v1/miner/start -H 'content-type: application/json' -d '{}'
```
