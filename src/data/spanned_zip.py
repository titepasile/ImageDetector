"""Read split ZIP archives (.z01, .z02, ..., .zip) in place, without joining them.

`readable_zip` in convert.py copies all volumes into one temporary file. For
multi-gigabyte datasets this module instead exposes the volumes as one virtual,
seekable file and keeps the patched end records in memory.
"""

from contextlib import contextmanager
import io
from pathlib import Path
import re
import struct
import zipfile

from .convert import normalize_split_end


class MissingVolumeError(OSError):
    pass


def declared_volumes(path):
    """Total number of volumes recorded in the end records of the final .zip."""
    with Path(path).open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 65557))
        tail = stream.read()
    index = tail.rfind(b"PK\x05\x06")
    if index < 0:
        raise ValueError(f"No ZIP end record in {path}")
    disk = struct.unpack_from("<4sH", tail, index)[1]
    if index >= 20 and tail[index - 20 : index - 16] == b"PK\x06\x07":
        return struct.unpack_from("<4sIQI", tail, index - 20)[3]
    return disk + 1


def split_volumes(path, allow_missing=False):
    """Return volume paths in order. Missing volumes are returned as None if allowed."""
    path = Path(path)
    parts = {
        int(p.suffix[2:]): p
        for p in path.parent.iterdir()
        if p.stem.lower() == path.stem.lower()
        and re.fullmatch(r"\.z\d+", p.suffix.lower())
    }
    count = max(parts, default=0)
    if allow_missing:
        count = max(count, declared_volumes(path) - 1)
    volumes = [parts.get(number) for number in range(1, count + 1)]
    if None in volumes and not allow_missing:
        raise ValueError(f"Missing split ZIP volumes for {path}")
    return [*volumes, path]


class SpannedFile(io.RawIOBase):
    """Concatenated read-only view of several files with an in-memory write overlay.

    A None path stands for a missing volume. Split volumes before the last one
    share one fixed size, so later offsets stay correct; reading inside a missing
    volume raises MissingVolumeError.
    """

    def __init__(self, paths):
        super().__init__()
        self.paths = [Path(p) if p is not None else None for p in paths]
        present = [p.stat().st_size for p in self.paths[:-1] if p is not None]
        if None in self.paths[:-1] and len(set(present)) != 1:
            raise ValueError("Cannot infer the size of missing split volumes")
        self.offsets, total = [], 0
        for path in self.paths:
            self.offsets.append(total)
            total += path.stat().st_size if path is not None else present[0]
        self.size = total
        self.position = 0
        self.handles = [None] * len(self.paths)
        self.overlay = {}  # absolute offset -> bytes

    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        self.position = max(0, base[whence] + offset)
        return self.position

    def _handle(self, index):
        if self.paths[index] is None:
            raise MissingVolumeError(f"Split ZIP volume {index + 1} is missing")
        if self.handles[index] is None:
            self.handles[index] = self.paths[index].open("rb")
        return self.handles[index]

    def _read_raw(self, start, size):
        chunks = []
        end = min(self.size, start + size)
        while start < end:
            index = max(i for i, o in enumerate(self.offsets) if o <= start)
            volume_end = (
                self.offsets[index + 1] if index + 1 < len(self.offsets) else self.size
            )
            handle = self._handle(index)
            handle.seek(start - self.offsets[index])
            count = min(end, volume_end) - start
            chunk = handle.read(count)
            if len(chunk) != count:
                raise OSError(f"Short read in {self.paths[index]}")
            chunks.append(chunk)
            start += count
        return b"".join(chunks)

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.position
        start = self.position
        data = bytearray(self._read_raw(start, size))
        for offset, patch in self.overlay.items():
            lo, hi = max(start, offset), min(start + len(data), offset + len(patch))
            if lo < hi:
                data[lo - start : hi - start] = patch[lo - offset : hi - offset]
        self.position += len(data)
        return bytes(data)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def write(self, data):
        self.overlay[self.position] = bytes(data)
        self.position += len(data)
        return len(data)

    def close(self):
        for handle in self.handles:
            if handle is not None:
                handle.close()
        super().close()


@contextmanager
def open_spanned_zip(path, allow_missing=False):
    """Open a split or single ZIP. Members can be read with archive.open().

    With allow_missing=True, members stored in absent volumes raise
    MissingVolumeError when read; all other members remain readable.
    """
    volumes = split_volumes(path, allow_missing)
    if len(volumes) == 1:
        with zipfile.ZipFile(path) as archive:
            yield archive
        return
    stream = SpannedFile(volumes)
    try:
        try:
            central_offset = normalize_split_end(stream, stream.offsets)
        except (struct.error, IndexError) as exc:
            raise ValueError("Truncated or invalid split ZIP end records") from exc
        with zipfile.ZipFile(stream) as archive:
            if archive.start_dir != central_offset:
                raise ValueError("Unexpected central directory offset in split ZIP")
            for member in archive.infolist():
                if member.volume >= len(stream.offsets):
                    raise ValueError("ZIP references a missing volume")
                member.header_offset += stream.offsets[member.volume]
            end = archive.start_dir
            for member in sorted(
                archive.infolist(), key=lambda m: m.header_offset, reverse=True
            ):
                member._end_offset = end
                end = member.header_offset
            yield archive
    finally:
        stream.close()
