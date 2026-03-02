# Example: Service Boot Idle -> Start

```bash
./seine --service --api-bind 127.0.0.1:9977
curl -s http://127.0.0.1:9977/v1/runtime/state
curl -s -X POST http://127.0.0.1:9977/v1/miner/start \
  -H 'content-type: application/json' \
  -d '{"mode":"pool","mining_address":"PpkFxY...","pool_url":"stratum+tcp://pool.example.com:3333","pool_worker":"rig-01"}'
```
