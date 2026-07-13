# Copyright 2026 Core Epoch LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import gc
import sys
import torch
import soundfile as sf
from mcp.server.fastmcp import FastMCP
from diffusers import StableAudioPipeline

# Initialize FastMCP server
mcp = FastMCP("mcp-soundfx")

MODEL_ID = "stabilityai/stable-audio-open-1.0"

# Optional sandbox root. Left unset, the server writes wherever the process can,
# which is what an agent authoring assets straight into a game project needs.
# Set MCP_SOUNDFX_OUTPUT_DIR and every generated file must resolve inside it.
OUTPUT_DIR_ENV = "MCP_SOUNDFX_OUTPUT_DIR"


def _resolve_output_path(output_path: str) -> str:
    """Canonicalize and validate a caller-supplied output path.

    This tool takes its destination from the caller, so an agent acting on a
    hostile prompt could otherwise be steered into writing bytes to any path the
    process can reach. Two checks bound that: the resolved path must name a .wav
    file, and when MCP_SOUNDFX_OUTPUT_DIR is set it must resolve inside that
    root. Raises ValueError naming the reason when either check fails.
    """
    if not output_path or not output_path.strip():
        raise ValueError("output_path must not be empty")

    resolved = os.path.realpath(os.path.expanduser(output_path.strip()))

    if os.path.splitext(resolved)[1].lower() != ".wav":
        raise ValueError(
            f"output_path must name a .wav file (got '{os.path.basename(resolved)}')"
        )

    root = os.environ.get(OUTPUT_DIR_ENV)
    if root:
        root = os.path.realpath(os.path.expanduser(root))
        try:
            inside = os.path.commonpath([resolved, root]) == root
        except ValueError:
            inside = False  # different drives on Windows have no common path
        if not inside:
            raise ValueError(
                f"output_path resolves outside {OUTPUT_DIR_ENV} ({root})"
            )

    return resolved


@mcp.tool()
def generate_sound(
    prompt: str,
    output_path: str,
    negative_prompt: str = "",
    duration_seconds: float = 5.0,
    steps: int = 100,
    seed: int = 42
) -> str:
    """
    Generate a high-quality sound effect using Stable Audio Open 1.0 running locally on CUDA GPU.

    This tool produces 44.1kHz stereo WAV files from text descriptions. It runs
    Stability AI's diffusion-based audio model entirely offline (after the initial
    model download), with no API calls, usage limits, or per-generation cost.

    IMPORTANT CONSTRAINTS:
    - Maximum duration is 47 seconds. Optimal range is 1–30 seconds.
    - output_path must name a .wav file. If MCP_SOUNDFX_OUTPUT_DIR is set in the
      environment, the path must also resolve inside that directory.
    - The model excels at sound effects, foley, ambient textures, and musical phrases.
    - It does NOT generate speech, vocals, singing, or dialogue.
    - Generation takes ~15–45 seconds depending on duration and step count.
    - The model loads into GPU VRAM (~2.5 GB in float16) and is cleaned up after each call.

    PROMPT ENGINEERING GUIDE:
    Write prompts as if describing audio for a sound designer. Be specific and descriptive.
    Good prompts include:
    - Physical material and action: "glass shattering on a tile floor"
    - Environment and space: "footsteps echoing in a large cathedral"
    - Tonal qualities: "a warm, resonant bell chime with a long decay"
    - Temporal structure: "a quick double-tap click followed by a soft whoosh"
    - Style references: "retro 8-bit arcade coin pickup sound"
    - Emotional quality: "an ominous low rumble building tension"

    Avoid vague prompts like "a sound" or "something cool". The more acoustically
    descriptive the prompt, the better the output quality.

    NEGATIVE PROMPT GUIDE:
    Use negative_prompt to steer away from unwanted audio artifacts:
    - "hiss, static, distortion, noise" — removes recording noise
    - "silence, quiet, muffled" — prevents dead air in the output
    - "music, melody, rhythm" — keeps output as pure SFX when you want foley
    - "reverb, echo" — produces dry/close-mic'd sounds
    - "harsh, loud, clipping" — tames aggressive transients

    PARAMETER TUNING:
    - duration_seconds: Match to the sound's natural length. UI clicks: 0.5–1.5s.
      Button hovers: 0.3–0.8s. Impacts/explosions: 2–5s. Ambience loops: 10–30s.
    - steps: 50 = fast drafts, 100 = production quality, 200 = maximum fidelity.
      Diminishing returns above 150. Use 50 for iteration, 100+ for final assets.
    - seed: Same seed + same prompt = deterministic output. Change seed to get
      variations of the same prompt. Use this to A/B test different generations.

    Args:
        prompt: Detailed text description of the sound to generate.
        output_path: Absolute path for the output file. Must end in .wav, and must
            resolve inside MCP_SOUNDFX_OUTPUT_DIR when that variable is set.
        negative_prompt: Text describing audio qualities to avoid in the output. Leave empty for no negative guidance.
        duration_seconds: The length of the generated sound in seconds (max 47.0, default 5.0).
        steps: Diffusion inference steps. 50=fast draft, 100=production, 200=max fidelity (default 100).
        seed: Random seed for reproducibility (default 42). Change to get variations.
    """
    # Validate the destination before spending 15–45 s of GPU time on a call
    # whose write would be refused anyway.
    try:
        abs_output_path = _resolve_output_path(output_path)
    except ValueError as e:
        message = f"Rejected output_path: {e}"
        print(message, file=sys.stderr)
        return message

    # Verify HF token or acceptance of terms
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import HfApi
            token = HfApi().token
        except Exception:
            pass

    if not token:
        print("Warning: HF_TOKEN environment variable is not set. Gated model download might fail if not cached.", file=sys.stderr)

    print(f"Starting generation for prompt: '{prompt}' (duration: {duration_seconds}s, steps: {steps})", file=sys.stderr)
    if negative_prompt:
        print(f"Negative prompt: '{negative_prompt}'", file=sys.stderr)

    pipe = None
    try:
        # Load pipeline in float16 to minimize VRAM usage
        print("Loading Stable Audio Open pipeline...", file=sys.stderr)
        pipe = StableAudioPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            use_safetensors=True,
            token=token
        )
        pipe = pipe.to("cuda")

        # Set generator seed
        generator = torch.Generator("cuda").manual_seed(seed)

        print("Running diffusion process...", file=sys.stderr)
        # Build inference kwargs
        inference_kwargs = dict(
            prompt=prompt,
            num_inference_steps=steps,
            audio_end_in_s=duration_seconds,
            num_waveforms_per_prompt=1,
            generator=generator
        )
        if negative_prompt:
            inference_kwargs["negative_prompt"] = negative_prompt

        audio = pipe(**inference_kwargs).audios

        # Format output
        output_data = audio[0].T.float().cpu().numpy()

        # Ensure directory exists. dirname is empty for a bare filename, and
        # os.makedirs("") raises, so only create when there is a parent.
        parent = os.path.dirname(abs_output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        print(f"Saving output to {abs_output_path}...", file=sys.stderr)
        sf.write(abs_output_path, output_data, pipe.vae.sampling_rate)

        return f"Successfully generated sound and saved to {abs_output_path}"

    except Exception as e:
        error_msg = f"Failed to generate sound: {str(e)}"
        print(error_msg, file=sys.stderr)
        return error_msg

    finally:
        # Clean up model weights so the GPU is free between generations
        print("Cleaning up GPU memory...", file=sys.stderr)
        if pipe is not None:
            del pipe
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    mcp.run()

