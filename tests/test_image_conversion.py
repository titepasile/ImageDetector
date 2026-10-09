import io
import json
from pathlib import Path
import shutil
import subprocess
import struct
import tarfile
import tempfile
import unittest
import zipfile

from PIL import Image
from torch.utils.data import DataLoader

from src.data.convert import convert_images
from src.data.streaming import StreamingImageDataset


class ImageConversionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "input"
        self.input.mkdir()
        self.output = self.input / "processed"

    def test_folders_zip_resize_errors_and_streaming_workers(self):
        nested = self.input / "nested"
        nested.mkdir()
        Image.new("RGBA", (45, 90), (255, 0, 0, 0)).save(nested / "alpha.png")
        Image.new("L", (200, 40), 100).save(nested / "gray.BMP")
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (80, 160), "green").save(nested / "oriented.jpg", exif=exif)
        payload = io.BytesIO()
        Image.new("RGB", (300, 20), "blue").save(payload, "JPEG")
        with zipfile.ZipFile(self.input / "images.zip", "w") as archive:
            archive.writestr("folder/image.jpg", payload.getvalue())
            archive.writestr("folder/notes.txt", "not an image")
            archive.writestr("folder/broken.png", b"broken")
        (self.input / "unfinished.z01.part").write_bytes(b"partial")
        (self.input / "orphan.z01").write_bytes(b"orphan")
        (self.input / "broken.zip").write_bytes(b"broken")
        manifest = convert_images(self.input, self.output, shard_size=2)
        self.assertEqual(manifest["images"], 4)
        self.assertEqual(len(manifest["shards"]), 2)
        self.assertEqual(manifest["failed_images"], 1)
        self.assertEqual(manifest["failed_archives"], 1)
        self.assertEqual(manifest["incomplete_archives"], 2)
        for shard in manifest["shards"]:
            with tarfile.open(self.output / shard["name"]) as archive:
                for member in archive:
                    if member.name.endswith(".png"):
                        with Image.open(archive.extractfile(member)) as image:
                            self.assertEqual(image.size, (128, 128))
                            self.assertEqual(image.mode, "RGB")
        dataset = StreamingImageDataset(self.output)
        samples = list(dataset)
        self.assertEqual(len(samples), 4)
        for tensor, metadata in samples:
            self.assertEqual(tuple(tensor.shape), (3, 128, 128))
            self.assertGreaterEqual(tensor.min().item(), 0)
            self.assertLessEqual(tensor.max().item(), 1)
        alpha = next(tensor for tensor, meta in samples if meta["source"].endswith("alpha.png"))
        self.assertTrue((alpha == 1).all())
        sources = [meta["source"] for _, meta in DataLoader(
            dataset, batch_size=None, num_workers=2, timeout=10)]
        self.assertEqual(len(sources), 4)
        self.assertEqual(len(set(sources)), 4)
        with self.assertRaisesRegex(ValueError, "empty"):
            convert_images(self.input, self.output)

    @unittest.skipUnless(shutil.which("zip"), "Info-ZIP is required")
    def test_real_split_zip(self):
        staging = self.root / "staging"
        staging.mkdir()
        Image.effect_noise((512, 512), 100).convert("RGB").save(staging / "noise.png")
        archive = self.input / "split.zip"
        subprocess.run(["zip", "-q", "-0", "-s", "64k", str(archive), "noise.png"],
                       cwd=staging, check=True)
        self.assertTrue(archive.with_suffix(".z01").exists())
        manifest = convert_images(self.input, self.output)
        self.assertEqual(manifest["images"], 1)
        self.assertEqual(manifest["failed_archives"], 0)
        self.assertEqual(manifest["incomplete_archives"], 0)
        samples = list(StreamingImageDataset(self.output))
        self.assertEqual(samples[0][1]["source"], "split.zip!noise.png")

    def test_empty_input_and_invalid_shard_size(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            convert_images(self.input, self.output, shard_size=0)
        manifest = convert_images(self.input, self.output)
        self.assertEqual(manifest["images"], 0)
        self.assertEqual(list(StreamingImageDataset(self.output)), [])
        self.assertEqual(json.loads((self.output / "manifest.json").read_text())["shards"], [])

    @unittest.skipUnless(shutil.which("zip"), "Info-ZIP is required for the fixture")
    def test_split_zip64_end_records(self):
        staging = self.root / "staging"
        staging.mkdir()
        Image.effect_noise((512, 512), 100).save(staging / "noise.png")
        path = self.input / "split.zip"
        subprocess.run(["zip", "-q", "-0", "-s", "64k", str(path), "noise.png"],
                       cwd=staging, check=True)
        data = path.read_bytes()
        position = len(data) - 22
        end = list(struct.unpack("<4s4H2IH", data[position:]))
        # Exercise ZIP64 end handling without allocating a multi-GB fixture.
        record = struct.pack("<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45,
                             end[1], end[2], end[3], end[4], end[5], end[6])
        locator = struct.pack("<4sIQI", b"PK\x06\x07", end[1], position, end[1] + 1)
        end[1:5] = [0xFFFF] * 4
        end[5:7] = [0xFFFFFFFF] * 2
        path.write_bytes(data[:position] + record + locator + struct.pack("<4s4H2IH", *end))
        manifest = convert_images(self.input, self.output)
        self.assertEqual(manifest["images"], 1)
        self.assertEqual(manifest["failed_archives"], 0)
        self.assertEqual(len(list(StreamingImageDataset(self.output))), 1)


if __name__ == "__main__":
    unittest.main()
