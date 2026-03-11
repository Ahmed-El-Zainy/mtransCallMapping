# transCallMapping

```
transCallMapping/
├── main.py          — FastAPI app (5 routes)
├── config.py        — Settings + both prompts + filename parser
├── app/
│   ├── transcriber.py  — Whisper transcription → [HH:MM:SS.mmm] format
│   ├── refiner.py      — Azure OpenAI Arabic + English refinement
│   └── writer.py       — Saves .txt × 3 + combined .json
├── .env.example
├── requirements.txt
└── README.md
```



# transCallMapping

> **Audio → Timestamped Transcript → Arabic + English refined versions**  
> FastAPI service for ELAraby Group call-centre recordings.

---

## Architecture

```
Audio file (upload)
        │
        ▼
 ┌─────────────┐
 │  Whisper    │  OpenAI whisper-1
 │  Transcribe │  → timestamped raw transcript
 └─────────────┘
        │
        ├─── filename parser ──► context_info (date / agent / topic)
        │
        ▼
 ┌─────────────────────┐        ┌─────────────────────┐
 │  Azure OpenAI       │        │  Azure OpenAI       │
 │  Arabic refinement  │        │  English refinement │
 │  (Egyptian dialect) │        │  (professional EN)  │
 └─────────────────────┘        └─────────────────────┘
        │                                │
        └──────────────┬─────────────────┘
                       ▼
              outputs/<stem>_<ts>_original.txt
              outputs/<stem>_<ts>_arabic.txt
              outputs/<stem>_<ts>_english.txt
              outputs/<stem>_<ts>_result.json
                       │
                       ▼
                  JSON response
```

---

## Quick Start

### 1. Clone & install

```bash
git clone <repo-url>
cd transCallMapping
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

`.env` keys:

| Key | Description |
|-----|-------------|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | `https://<resource>.openai.azure.com/` |
| `AZURE_OPENAI_API_VERSION` | e.g. `2024-02-01` |
| `AZURE_OPENAI_DEPLOYMENT` | Deployment name (e.g. `gpt-4o-mini`) |
| `OPENAI_API_KEY` | OpenAI API key (for Whisper transcription) |
| `OUTPUT_DIR` | Where to save result files (default: `./outputs`) |

### 3. Run

```bash
uvicorn main:app --reload --port 8000
```

Interactive docs: **http://localhost:8000/docs**

---

## API Reference

### `POST /process` — Full pipeline (recommended)

Upload an audio file → get transcription + Arabic + English refinements + saved files.

```bash
curl -X POST http://localhost:8000/process \
  -F "audio=@20240315_AGT001_CUST4892_washing_machine_complaint.mp3"
```

**Response (JSON):**
```json
{
  "filename": "20240315_AGT001_CUST4892_washing_machine_complaint.mp3",
  "context_info": "Call Date: 2024-03-15\nAgent ID: AGT001\nCustomer ID: CUST4892\nCall Topic / Notes: Washing Machine Complaint",
  "processed_at": "2024-03-15T10:30:00Z",
  "original_transcription": "[00:00:00.000] آلو، معاك خدمة عملاء العربي...",
  "arabic_refined": "[00:00:00.000] Agent: أهلاً وسهلاً، معك خدمة عملاء العربي جروب...",
  "english_refined": "[00:00:00.000] Agent: Hello, you've reached ELAraby Group customer service...",
  "output_files": {
    "original": "./outputs/20240315_..._original.txt",
    "arabic":   "./outputs/20240315_..._arabic.txt",
    "english":  "./outputs/20240315_..._english.txt",
    "json":     "./outputs/20240315_..._result.json"
  }
}
```

---

### `POST /transcribe` — Transcription only

Transcribe audio without LLM refinement (faster, cheaper).

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@call.mp3"
```

---

### `GET /outputs` — List output files

```bash
curl http://localhost:8000/outputs
```

---

### `GET /outputs/{filename}` — Download output file

```bash
curl http://localhost:8000/outputs/call_20240315T103000Z_arabic.txt -o arabic.txt
```

---

### `GET /health`

```bash
curl http://localhost:8000/health
# → {"status": "ok", "service": "transCallMapping"}
```

---

## Filename → `context_info` Detection

The filename is automatically parsed to extract structured context injected into the LLM prompt.

| Filename pattern | Extracted context |
|---|---|
| `20240315_AGT001_CUST4892_complaint.mp3` | Date · Agent ID · Customer ID · Topic |
| `call-2024-03-15-agent-ahmed-refund.wav` | Date · Topic words |
| `IVR_delivery_issue_2024.mp3` | Topic · Date |
| `random_name.wav` | Source file fallback |

Supported separators: `_` and `-`.

---

## Project Structure

```
transCallMapping/
├── main.py              # FastAPI app & routes
├── config.py            # Settings, prompts, filename parser
├── requirements.txt
├── .env.example
├── app/
│   ├── __init__.py
│   ├── transcriber.py   # Whisper transcription
│   ├── refiner.py       # Azure OpenAI Arabic + English refinement
│   └── writer.py        # Save .txt / .json output files
└── outputs/             # Generated files land here
```

---

## Supported Audio Formats

`.mp3` · `.wav` · `.m4a` · `.ogg` · `.flac` · `.webm` · `.mp4`

---

## Notes for Prompt Engineers

The two prompts live in `config.py` as:
- `ARABIC_USER_PROMPT_TEMPLATE` — Egyptian Arabic refinement
- `ENGLISH_USER_PROMPT_TEMPLATE` — Professional English refinement

Both accept `{context_info}` and `{original_transcription}` placeholders.  
Edit them freely; no code changes needed elsewhere.