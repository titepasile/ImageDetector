"""PyTorch reader for converted tar shards; no WebDataset dependency needed."""

import io
import json
from pathlib import Path
import tarfile

from PIL import Image
import torch
from torch.utils.data import IterableDataset, get_worker_info


class StreamingImageDataset(IterableDataset):
    """Yield (float32 CHW image in [0,1], metadata) with bounded memory.

    Shards are divided between DataLoader workers and distributed ranks.
    No labels are inferred; use metadata['source'] to apply your label rules.
    """

    def __init__(self, directory, transform=None):
        super().__init__()
        self.directory = Path(directory)
        manifest = json.loads((self.directory / "manifest.json").read_text())
        self.shards = [self.directory / shard["name"] for shard in manifest["shards"]]
        self.transform = transform
        # Capture rank before spawning DataLoader workers.
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        self.rank = torch.distributed.get_rank() if distributed else 0
        self.world_size = torch.distributed.get_world_size() if distributed else 1

    def __iter__(self):
        worker = get_worker_info()
        workers, worker_id = (worker.num_workers, worker.id) if worker else (1, 0)
        offset = self.rank * workers + worker_id
        for path in self.shards[offset::self.world_size * workers]:
            with tarfile.open(path, "r|*") as archive:
                pending = None
                key = None
                for member in archive:
                    if not member.isfile():
                        continue
                    stream = archive.extractfile(member)
                    if member.name.endswith(".png"):
                        with Image.open(io.BytesIO(stream.read())) as image:
                            image = image.convert("RGB")
                            if self.transform is not None:
                                pending = self.transform(image)
                            else:
                                pending = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
                                pending = pending.reshape(image.height, image.width, 3).permute(2, 0, 1).float().div_(255)
                        key = member.name[:-4]
                    elif member.name.endswith(".json") and member.name[:-5] == key:
                        yield pending, json.loads(stream.read())
                        pending, key = None, None
