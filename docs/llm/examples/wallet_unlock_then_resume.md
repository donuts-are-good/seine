# Example: Wallet Unlock Then Resume

```bash
curl -s -X POST http://127.0.0.1:9977/v1/wallet/unlock \
  -H 'content-type: application/json' \
  -d '{"password":"<wallet-password>"}'
curl -s -X POST http://127.0.0.1:9977/v1/miner/restart -H 'content-type: application/json' -d '{}'
```
