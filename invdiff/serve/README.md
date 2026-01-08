# VisDiff API Servers

## Design Choices

1. All LLMs/VLMs/CLIPs serve as API with cache enabled, because loading a LLM/VLM/CLIP is expensive and we never modify them.
2. LLM functions in `utils_llm.py`, VLM functions in `utils_vlm.py`, CLIP functions in `utils_clip.py`, and others in `utils_general.py`.
3. Write unit tests to understand major functions.

## LLM Server Configuration

1. Set up OpenAI API key: `export OPENAI_API_KEY='[your key]'`

## CLIP Server Configuration

1. Pip install environments: `pip install open-clip-torch flask`
2. Configure global variables in `global_vars.py`
3. Run `python invdiff/serve/clip_server.py`
4. Run `python -m invdiff.serve.utils_clip` to test the CLIP.

## VLM Server Configuration

### For BLIP-2

1. Create virtual environment and install deps from [blip-requirements.txt](/InvDiff/invdiff/captioners/blip/requirements.txt)
2. (Optional) Configure global variables in `global_vars.py`
3. Run `python invdiff/serve/vlm_server_blip.py`. It takes a while to load the VLM, especially the first time to download the VLM. (Note: concurrency is disabled as it surprisingly leads to worse GPU utilization)
4. Run `python -m invdiff.serve.utils_vlm` to test BLIP.
5. Note: We need to create a separate virtual environment for this, else there will be dependency conflicts.

### For GIT (Generative Image-to-text)

1. Update `base.yaml`: replace `blip` with `git` everywhere in the file.
2. Run `python invdiff/serve/vlm_server_git.py`.
3. Run `python -m invdiff.serve.utils_vlm` to test GIT.
