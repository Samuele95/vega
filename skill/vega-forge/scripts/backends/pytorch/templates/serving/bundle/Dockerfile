# {title} as {what}, CPU only.
# Build:  docker build -t {dist} .
FROM {base_image}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    {env}_BUNDLE=/app {env}_TORCH_THREADS=2
WORKDIR /app
# the CPU build of the torch version the bundle was verified with; the start-up self-test
# refuses to serve if this build does not reproduce the stored examples
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch=={torch}
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt && useradd --create-home --uid 10001 {cmd}
# the bundle stays owned by root and readable by all: the service reads its weights, recipe and
# examples and never writes them, so the user it runs as cannot alter what it serves. A showcase
# /vega-show built into showcase/ ships with it (.dockerignore keeps the slot and leaves out its
# previous copy, an unfinished build, and its smoke and working directories)
COPY . .
USER {cmd}
{runtime}
