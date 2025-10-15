import os
import asyncio
from typing import List, Tuple, Optional

import aiohttp
import aiofiles
import torch
from PIL import Image


def get_device() -> torch.device:

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_jinaclip(model_name: str = "jinaai/jina-clip-v2"):

    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    device = get_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device


async def _download_image(session: aiohttp.ClientSession, url: str, save_path: str, retries: int = 2) -> Optional[str]:

    if os.path.exists(save_path):
        return save_path
    for _ in range(retries + 1):
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    async with aiofiles.open(save_path, "wb") as f:
                        await f.write(await response.read())
                    return save_path
        except Exception:
            continue
    return None


async def download_images(urls: List[str], cache_dir: str) -> List[Optional[str]]:

    os.makedirs(cache_dir, exist_ok=True)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for idx, url in enumerate(urls):
            target_path = os.path.join(cache_dir, f"{idx}.jpg")
            tasks.append(_download_image(session, url, target_path))
        return await asyncio.gather(*tasks)


@torch.no_grad()
def encode_texts(model, texts: List[str], device: torch.device, batch_size: int = 128) -> torch.Tensor:

    embeddings: List[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        emb = model.encode_text(batch_texts, convert_to_tensor=True, device=str(device))
        embeddings.append(emb.to("cpu"))
    return torch.cat(embeddings, dim=0)


@torch.no_grad()
def encode_images(model, image_paths: List[Optional[str]], device: torch.device, batch_size: int = 32, placeholder_path: Optional[str] = None):

    def load_rgb(path: Optional[str]) -> Image.Image:
        if path and os.path.exists(path):
            return Image.open(path).convert("RGB")
        if placeholder_path and os.path.exists(placeholder_path):
            return Image.open(placeholder_path).convert("RGB")
        return Image.new("RGB", (224, 224), color=(0, 0, 0))

    batch_images: List[Image.Image] = []
    batch_indices: List[int] = []
    for idx, path in enumerate(image_paths):
        try:
            img = load_rgb(path)
        except Exception:
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        batch_images.append(img)
        batch_indices.append(idx)

        if len(batch_images) == batch_size or idx == len(image_paths) - 1:
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16):
                batch_emb = model.encode_image(batch_images)
            yield batch_indices, torch.as_tensor(batch_emb, dtype=torch.float32)
            batch_images, batch_indices = [], []


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:

    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


def fuse_embeddings(text_embeddings: torch.Tensor, image_embeddings: torch.Tensor, method: str = "concat") -> torch.Tensor:

    text_embeddings = l2_normalize(text_embeddings)
    image_embeddings = l2_normalize(image_embeddings)
    if method == "concat":
        fused = torch.cat([text_embeddings, image_embeddings], dim=-1)
    else:
        fused = text_embeddings + image_embeddings
    return l2_normalize(fused)


async def build_fused_embeddings_from_urls(
    texts: List[str],
    image_urls: List[str],
    cache_dir: str = "cache_images",
    model_name: str = "jinaai/jina-clip-v2",
    fuse_method: str = "concat",
) -> torch.Tensor:

    _, model, device = load_jinaclip(model_name)
    os.makedirs(cache_dir, exist_ok=True)
    placeholder_path = os.path.join(cache_dir, "placeholder.jpg")
    if not os.path.exists(placeholder_path):
        Image.new("RGB", (224, 224), color=(0, 0, 0)).save(placeholder_path)

    image_paths = await download_images(image_urls, cache_dir)
    text_emb = encode_texts(model, texts, device=device, batch_size=128)

    fused_embeddings: Optional[torch.Tensor] = None
    for batch_indices, img_emb in encode_images(model, image_paths, device=device, batch_size=32, placeholder_path=placeholder_path):
        if fused_embeddings is None:
            fused_embeddings = torch.zeros((len(texts), text_emb.shape[1] + img_emb.shape[1]), dtype=torch.float32)
        fused_embeddings[batch_indices] = fuse_embeddings(text_emb[batch_indices], img_emb, method=fuse_method)

    return fused_embeddings if fused_embeddings is not None else torch.empty((0,))


