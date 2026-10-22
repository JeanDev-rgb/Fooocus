import os
from urllib.parse import urlparse, urlsplit, urlunsplit, parse_qsl, urlencode
from typing import Optional


def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
) -> str:
    """Download a file from a URL, using the local file if it already exists.

    For Civitai URLs, CIVITAI_TOKEN can be supplied through the environment.
    The token is never written to the repository or printed to the console.
    """
    domain = os.environ.get("HF_MIRROR", "https://huggingface.co").rstrip('/')
    url = str.replace(url, "https://huggingface.co", domain, 1)

    if urlparse(url).netloc.lower().endswith("civitai.com"):
        token = os.environ.get("CIVITAI_TOKEN", "").strip()
        if token:
            parts = urlsplit(url)
            query = dict(parse_qsl(parts.query, keep_blank_values=True))
            query["token"] = token
            url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))

    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
        if not file_name or file_name == "models":
            file_name = "civitai_model.safetensors"

    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        # Do not print the full URL: it may contain an access token.
        safe_url = url.split("?", 1)[0]
        print(f'Downloading: "{safe_url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress)
    return cached_file
