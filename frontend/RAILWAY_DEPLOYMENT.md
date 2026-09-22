# Railway frontend deployment

The frontend needs a reachable FastAPI service for `/api/analyze`, `/api/documents/scan`, and the other detection routes.

Set this Railway variable before deploying the frontend:

```text
VITE_API_BASE_URL=https://<public-backend-service-domain>
```

The value must point directly to the FastAPI service, not another frontend domain. After changing it, trigger a new deployment because Vite embeds `VITE_*` variables during the build.

Verify the backend before testing the UI:

```text
https://<public-backend-service-domain>/health
```

It must return JSON with `"status":"healthy"`. If `/api/health` on the Railway frontend returns `502`, the backend service URL or Railway proxy is unavailable; rule and semantic detector fields cannot be displayed until that connection is fixed.