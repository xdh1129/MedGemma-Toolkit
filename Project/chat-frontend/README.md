# MedGemma Chat Frontend

This package contains the React client that powers the MedGemma chat experience. It is a Vite application that talks to the FastAPI backend through the `/api` reverse proxy provided by Nginx (or whatever value you pass through `VITE_API_BASE_URL`).

## Local development

```bash
cd chat-frontend
npm install
npm run dev
```

By default the app proxies requests to `/api`. You can point it at a different backend by creating a `.env` file with:

```bash
VITE_API_BASE_URL=http://localhost:8000/api
```

## Production build

```bash
npm run build
npm run preview
```
