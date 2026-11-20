"""Build a wheel with a stable archive layout across compression-library versions."""

import argparse
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_STORED, ZipFile, ZipInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as temporary:
        subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", temporary],
            check=True,
        )
        artifact, = Path(temporary).glob("*.whl")
        with ZipFile(artifact) as built, ZipFile(args.outdir / artifact.name, "w") as release:
            for entry in sorted(built.infolist(), key=lambda item: item.filename):
                info = ZipInfo(entry.filename, date_time=(2020, 2, 2, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                release.writestr(info, built.read(entry), compress_type=ZIP_STORED)


if __name__ == "__main__":
    main()