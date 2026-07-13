# mcp-soundfx

**Local text-to-SFX generation for AI agents.** An [MCP](https://modelcontextprotocol.io) server that runs Stability AI's [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) on your own NVIDIA GPU, so any MCP-capable agent (Claude Code, Claude Desktop, etc.) can generate 44.1 kHz stereo WAV sound effects from text prompts. After the initial model download it runs fully offline, with no API costs or usage limits.

## What it does

Exposes one tool, `generate_sound`:

| Parameter | Description |
|---|---|
| `prompt` | Sound description, written like a brief to a sound designer ("glass shattering on a tile floor", "retro 8-bit coin pickup") |
| `output_path` | Absolute path for the output file; must name a `.wav` |
| `negative_prompt` | Qualities to steer away from ("hiss, static, distortion") |
| `duration_seconds` | Up to 47 s (UI clicks: 0.5–1.5 s, impacts: 2–5 s, ambience: 10–30 s) |
| `steps` | 50 = fast draft, 100 = production, 200 = max fidelity |
| `seed` | Deterministic per (prompt, seed) — change to audition variations |

The model generates SFX, foley, ambience, and musical phrases; it does not generate speech or vocals. The pipeline loads in float16 (~2.5 GB VRAM) and is torn down after every call, so the GPU is free between generations.

## Setup

**Requirements:** Windows/Linux, Python 3.10+, NVIDIA GPU with ~3 GB free VRAM, CUDA-capable torch.

1. **Accept the gated model license** (one time): log into Hugging Face and accept the terms on the [stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) page, then create a **Read** token at [Token Settings](https://huggingface.co/settings/tokens).

2. **Install:**

   ```bash
   cd mcp-soundfx
   python -m venv .venv
   .venv\Scripts\activate          # Windows (source .venv/bin/activate on Linux)
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

3. **Register the server** with your MCP client. For Claude Code / Claude Desktop:

   ```json
   {
     "mcpServers": {
       "mcp-soundfx": {
         "command": "powershell.exe",
         "args": ["-ExecutionPolicy", "Bypass", "-File", "C:\\path\\to\\mcp-soundfx\\run.ps1"],
         "env": { "HF_TOKEN": "hf_your_read_token" }
       }
     }
   }
   ```

   The launch scripts use `.venv` next to the script by default; set `MCP_SOUNDFX_VENV` to point elsewhere.

4. **Smoke test** (optional, ~30 s on first run after model download):

   ```bash
   python test_generation.py
   ```

The first generation downloads the model weights (~2.5 GB) to your Hugging Face cache; everything after that is offline.

## Output paths

The resolved output path must name a `.wav` file. Set `MCP_SOUNDFX_OUTPUT_DIR` to confine every generation to one directory; unset, the server writes wherever the caller points it.

## License

The **server code** is [Apache License 2.0](LICENSE) © 2026 [Core Epoch LLC](https://coreepoch.dev).

The **model** (Stable Audio Open 1.0) is gated and separately licensed by Stability AI under the [Stability AI Community License](https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE.md) — free for research, non-commercial, and commercial use by entities under $1M annual revenue; you accept those terms directly with Stability AI when you unlock the model, and generated outputs are yours. This project is not affiliated with or endorsed by Stability AI.
