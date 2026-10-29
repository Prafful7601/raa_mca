# Certified Numeric Verification — live demo (Vercel)

This folder is a self-contained, deployable slice of the main project: the
interactive table-fact-checking demo (`index.html` + a FastAPI backend under
`api/`). It is **not** the research artifact itself — the paper, evaluation
scripts, and the 12k-claim NumNear-TabFact results live one level up in the
main repo and are not deployable to Vercel (they're not a web app).

## What's different from the original `api/server.py` / `frontend/index.html`

- Runs on **Gemini 3.5 Flash-Lite** only (not Ollama/Anthropic), since a
  serverless deployment can't reach your local Ollama instance, and Gemini
  is the model this project's real calibration numbers were computed on.
- All file writes go to `/tmp` (Vercel's deployed filesystem is read-only
  everywhere else, and `/tmp` itself doesn't persist across cold starts or
  across concurrent instances). The review queue (`/tmp/.../review_queue.xlsx`)
  and LLM response cache are therefore best-effort and ephemeral here —
  fine for a demo, not a substitute for the real pipeline's persistent
  `results/` and `cache/`.
- The calibrated conformal thresholds are bundled read-only at
  `calibration/conformal_thresholds.json` (copied from
  `results/tables/gemini_conformal_thresholds.json` in the main repo) rather
  than read from a results directory produced by the offline eval.
- The response includes **both** the raw majority-vote verdict and whether
  it clears the calibrated threshold (`certified`), instead of collapsing
  an uncertified claim into a bare `ABSTAIN`. Note: at low alpha
  (0.02–0.10) the real calibrated threshold for single-signal execution
  agreement is close to 1.0, so most claims will show as "below threshold →
  review" even when the raw verdict is correct — that's the paper's actual
  finding (within-method agreement alone is too coarse to certify strict
  error rates), not a bug in this demo.
- A best-effort per-IP rate limit (8 calls/min) and input-size caps (200
  rows, 40 columns, 20k chars of CSV, 500-char claims) guard against
  runaway spend on your Gemini key. The limiter resets on cold start and
  isn't shared across instances — it deters casual abuse, it does not
  replace setting a spend cap/alert on the API key itself at
  https://aistudio.google.com/ (recommended before sharing the URL widely).
  For real rate limiting, enable a Vercel Firewall rule from the project
  dashboard (Settings → Firewall).

## Deploy

1. `cd vercel_deploy`
2. Push this folder to its own GitHub repo (or a subfolder of one — if it's
   a subfolder, set that as the Vercel project's "Root Directory").
3. In the Vercel dashboard: New Project → import the repo → set the
   environment variable `GEMINI_API_KEY` (Project Settings → Environment
   Variables) to a real key from https://aistudio.google.com/apikey.
4. Deploy. Vercel auto-detects the Python function in `api/index.py` via
   `requirements.txt` and serves `index.html` as the static root.
5. Visit the deployed URL — the page calls its own `/api/health` and
   `/api/verify` (relative paths, same origin), nothing to configure.

### Local dev

```bash
npm i -g vercel   # if you don't already have the CLI
cp .env.example .env   # fill in your real GEMINI_API_KEY
vercel dev
```

## Known limitations of this deployment specifically

- No persistence: the review queue and cache vanish on every cold start.
  If you need a real review queue, point `append_review_row` at a database
  or external store instead of `/tmp`.
- Single model, fixed K=5, fixed temperature — matches the paper's main
  run; not configurable from the UI by design, to keep the demo's behavior
  traceable back to a specific calibrated result.
- The rate limiter and size caps are soft, in-memory, per-instance
  safeguards, not a substitute for API-key-level spend controls.
