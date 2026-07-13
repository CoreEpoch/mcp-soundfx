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
import sys

# Add current folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcp_soundfx import generate_sound

if __name__ == "__main__":
    # Ensure HF_TOKEN is configured
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Please set the HF_TOKEN environment variable before running this test script.", file=sys.stderr)
        print("Example: $env:HF_TOKEN = 'your_huggingface_read_token'", file=sys.stderr)
        sys.exit(1)

    print("Starting test sound generation...", flush=True)
    
    # Generate a simple 2-second bubble sound
    prompt = "a single water bubble pop sound effect"
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_pop.wav")
    
    result = generate_sound(
        prompt=prompt,
        output_path=output_file,
        duration_seconds=2.0,
        steps=50,  # few steps for quick test
        seed=123
    )
    
    print(f"Result: {result}", flush=True)
    if os.path.exists(output_file):
        print(f"SUCCESS: Generated file found at {output_file} ({os.path.getsize(output_file)} bytes)", flush=True)
    else:
        print("FAILURE: Generated file not found.", file=sys.stderr)
