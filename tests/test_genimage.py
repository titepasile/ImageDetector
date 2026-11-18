import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image

from src.data.dataset import read_manifest
from src.data.genimage import build_genimage_manifest, estimate_quality
from src.data.spanned_zip import (
    MissingVolumeError,
    SpannedFile,
    declared_volumes,
    open_spanned_zip,
    split_volumes,
)


def encoded(image, format, **options):
    buffer = io.BytesIO()
    image.save(buffer, format, **options)
    return buffer.getvalue()


class SpannedZipTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_reads_across_volumes_with_write_overlay(self):
        paths = []
        for index, data in enumerate((b"abc", b"defg", b"hi")):
            path = self.root / f"part{index}"
            path.write_bytes(data)
            paths.append(path)
        stream = SpannedFile(paths)
        self.addCleanup(stream.close)
        self.assertEqual(stream.offsets, [0, 3, 7])
        stream.seek(2)
        self.assertEqual(stream.read(4), b"cdef")
        stream.seek(4)
        stream.write(b"XY")
        stream.seek(0)
        self.assertEqual(stream.read(), b"abcdXYghi")
        self.assertEqual(paths[1].read_bytes(), b"defg")  # Files stay unchanged.
        stream.seek(-1, 2)
        self.assertEqual(stream.read(5), b"i")

    def test_single_volume_archive(self):
        path = self.root / "single.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("a.txt", "hello")
        self.assertEqual(declared_volumes(path), 1)
        with open_spanned_zip(path, allow_missing=True) as archive:
            self.assertEqual(archive.read("a.txt"), b"hello")

    def test_missing_volume_keeps_offsets_and_fails_only_inside_it(self):
        first, last = self.root / "part0", self.root / "part2"
        first.write_bytes(b"abcd")
        last.write_bytes(b"ij")
        stream = SpannedFile([first, None, last])
        self.addCleanup(stream.close)
        self.assertEqual(stream.offsets, [0, 4, 8])
        stream.seek(8)
        self.assertEqual(stream.read(), b"ij")
        stream.seek(2)
        with self.assertRaises(MissingVolumeError):
            stream.read(4)
        self.assertEqual(split_volumes(last), [last])  # No .zNN siblings.


class GenImageBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_debiased_manifest(self):
        archive = self.root / "imagenet_ai_0508_toy.zip"
        with zipfile.ZipFile(archive, "w") as output:
            for source in ("train", "val"):
                for index in range(4):
                    real = Image.effect_noise((300 + 8 * index, 280), 60).convert("RGB")
                    output.writestr(
                        f"imagenet_ai_0508_toy/{source}/nature/n0_{source}{index}.JPEG",
                        encoded(real, "JPEG", quality=90, subsampling=0),
                    )
                    fake = Image.effect_noise((256, 256), 60).convert("RGBA")
                    output.writestr(
                        f"imagenet_ai_0508_toy/{source}/ai/{index}_toy_{source}.PNG",
                        encoded(fake, "PNG"),
                    )
                # Unusable real images: too small and grayscale.
                output.writestr(
                    f"imagenet_ai_0508_toy/{source}/nature/small.JPEG",
                    encoded(Image.new("RGB", (200, 300)), "JPEG"),
                )
                output.writestr(
                    f"imagenet_ai_0508_toy/{source}/nature/gray.JPEG",
                    encoded(Image.new("L", (300, 300)), "JPEG"),
                )
        output_dir = self.root / "built"
        counts = {"train": ("train", 2), "val_id": ("train", 1), "test_id": ("val", 2)}
        summary = build_genimage_manifest(archive, output_dir, counts, seed=3)
        records = read_manifest(output_dir / "samples.jsonl")
        self.assertEqual(len(records), 10)
        self.assertEqual(summary["stats"]["test_id_ai"], 2)
        members = [r["source_member"] for r in records]
        self.assertEqual(len(members), len(set(members)))
        self.assertFalse(any("small" in m or "gray" in m for m in members))
        for record in records:
            with Image.open(record["path"]) as image:
                self.assertEqual((image.format, image.mode, image.size), ("PNG", "RGB", (256, 256)))
            self.assertEqual(record["jpeg_quality"], 90)
            self.assertEqual(record["jpeg_subsampling"], "4:4:4")
            if record["label"]:
                self.assertEqual(record["generator_family"], "toy")
        self.assertEqual(json.loads((output_dir / "build.json").read_text())["seed"], 3)
        with self.assertRaises(FileExistsError):
            build_genimage_manifest(archive, output_dir, counts)
        # A lower native size crops real images smaller; generated 256 images no
        # longer match it and are skipped. output_size resizes both classes alike.
        resized = self.root / "resized"
        summary = build_genimage_manifest(
            archive, resized, {"test_id": ("val", 1)}, size=256, output_size=128
        )
        for record in read_manifest(resized / "samples.jsonl"):
            with Image.open(record["path"]) as image:
                self.assertEqual(image.size, (128, 128))
        self.assertEqual(summary["output_size"], 128)

    def test_quality_estimate(self):
        image = Image.effect_noise((64, 64), 50).convert("RGB")
        for quality in (50, 75, 96):
            with Image.open(io.BytesIO(encoded(image, "JPEG", quality=quality))) as jpeg:
                self.assertEqual(estimate_quality(jpeg), quality)


if __name__ == "__main__":
    unittest.main()
