# Example: Mixed CPU/NVIDIA with Progress Stream

```bash
curl -N http://127.0.0.1:9977/v1/events/stream
```

Watch for `nvidia.init.progress`, `backend.phase`, and `autotune.progress` events while startup continues.
