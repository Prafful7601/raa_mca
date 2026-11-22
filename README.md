# Certified Numeric Verification — live demo (Vercel)

This folder is a self-contained, deployable slice of the main project: the
interactive table-fact-checking demo (`public/index.html` + a FastAPI backend
under `api/`). It is **not** the research artifact itself — the paper,
evaluation scripts, and the 12k-claim NumNear-TabFact results live one level
up in the main repo and are not deployable to Vercel (they're not a web app).

## What's different from the original `api/server.py` / `frontend/index.html`

- Runs on **Groq's free tier** (`qwen/qwen3.8-27b`) by default, not Gemini.
  The project's Gemini billing project ran out of prepaid credit, and a
  since-tried Meta Model API key hit `billing_not_configured` on its team —
  Groq's free tier was the option that actually authenticated and served
  real completions with no billing error. Gemini (`gemini-3.5-flash-lite`)
  is still configured as a second available model in `config.yaml`, so
  flipping `models.default` back is a one-line change once Gemini has
  credit again (or any other provider — `src/llm_client.py` just needs a
  backend class per provider, already true for `anthropic`/`ollama` too).
- **The bundled conformal thresholds are only valid for `gemini-3.5-flash-lite`**
  (they were fit to that model's specific agreement-score/error-rate
  relationship). Running a different default model, this deployment never
  claims `"certified": true` — it still runs the full K=5 execution-agreement
  pipeline and shows the raw verdict + agreement score, but every response's
  `reason` says calibration doesn't apply, and the frontend labels it
  "not certified → review" rather than silently applying a threshold that
  has no statistical basis for this model. See `CALIBRATED_MODEL` in
  `api/index.py` if you switch the default back to Gemini.
- The frontend (`public/index.html`) is served as a plain **static file**,
  not through the Python function. An earlier version tried serving it from
  inside FastAPI (`GET /`), but Vercel's Python runtime sets the ASGI
  request path to the rewrite's literal destination string rather than the
  real incoming path, so every route looked identical to the app (confirmed
  by temporarily echoing `request.url.path` back in a 404 handler). Static
  files in `public/` sidestep the function entirely and don't hit that bug.
  The API routes that *do* go through the function (`/api/health`,
  `/api/verify`) work around the same issue with a small ASGI middleware
  (`RestoreRewrittenPath` in `api/index.py`) that recovers the real sub-path
  from a `?__path=$1` query param `vercel.json`'s rewrite appends — see the
  comments on that class and in `vercel.json` before changing either.
- All file writes go to `/tmp` (Vercel's deployed filesystem is read-only
  everywhere else, and `/tmp` itself doesn't persist across cold starts or
  across concurrent instances). The review queue (`/tmp/.../review_queue.xlsx`)
  and LLM response cache are therefore best-effort and ephemeral here —
  fine for a demo, not a substitute for the real pipeline's persistent
  `results/` and `cache/`.
- A best-effort per-IP rate limit (8 calls/min) and input-size caps (200
  rows, 40 columns, 20k chars of CSV, 500-char claims) guard against
  runaway spend/quota use. The limiter resets on cold start and isn't
  shared across instances — it deters casual abuse, it does not replace
  setting a spend cap/alert on the provider's own dashboard (AI Studio for
  Gemini, Groq Console for Groq) before sharing the URL widely. For real
  rate limiting, enable a Vercel Firewall rule from the project dashboard
  (Settings → Firewall).

## Deploy

1. `cd vercel_deploy`
2. Push this folder to its own GitHub repo (or a subfolder of one — if it's
   a subfolder, set that as the Vercel project's "Root Directory"). Make
   sure the Vercel GitHub App actually has access to the repo (Settings →
   Integrations on GitHub) — a repo created after the app was installed
   with "only select repositories" won't auto-deploy on push otherwise.
3. In the Vercel dashboard: New Project → import the repo → set the
   environment variable `GROQ_API_KEY` (Project Settings → Environment
   Variables) to a real key from https://console.groq.com/keys. Add
   `GEMINI_API_KEY` too if you want Gemini available as a fallback/manual
   switch.
4. Deploy. Vercel auto-detects the Python function in `api/index.py` via
   `requirements.txt`, and serves `public/index.html` as the static root.
5. Visit the deployed URL — the page calls its own `/api/health` and
   `/api/verify` (relative paths, same origin), nothing to configure.

### Local dev

```bash
npm i -g vercel   # if you don't already have the CLI
cp .env.example .env   # fill in your real GROQ_API_KEY (and/or GEMINI_API_KEY)
vercel dev
```

Or just run the FastAPI app directly with uvicorn for quick iteration:

```bash
GROQ_API_KEY=... uvicorn api.index:app --port 8099 --reload
```

## Known limitations of this deployment specifically

- No persistence: the review queue and cache vanish on every cold start.
  If you need a real review queue, point `append_review_row` at a database
  or external store instead of `/tmp`.
- Groq's free tier has a fairly tight per-minute token budget (observed:
  ~8000 tokens/min on a fresh key). One `/verify` call's K=5 sampled checks
  can use most of that in a single request, so back-to-back demo clicks
  within the same minute may hit a 429 that the client retries with
  backoff (already handled by `LLMClient`), just with added latency —
  not a bug, just the free tier's actual ceiling.
- Fixed K=5, fixed temperature — matches the paper's main run; not
  configurable from the UI by design, to keep the demo's behavior
  traceable back to a specific run.
- The rate limiter and size caps are soft, in-memory, per-instance
  safeguards, not a substitute for provider-side spend/quota controls.
