## Deploy Option B on Render

You will deploy TWO web services:
1) API (FastAPI) from /api (port 8000)
2) WEB (Next.js) from /web (port 3000)

### API service
- Environment: Docker
- Root directory: api
- Port: 8000

### WEB service
- Environment: Docker
- Root directory: web
- Port: 3000
- Add env var:
  NEXT_PUBLIC_API_BASE = https://<your-api-service>.onrender.com
