"""원본 MAT와 임시 파일을 제외한 독립 GitHub 저장소용 ZIP을 만듭니다."""

from pathlib import Path
import hashlib
import json
import zipfile

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_NAME = "battery-cycle-life-prediction"


def package():
    files = [PROJECT_DIR / ".gitignore", PROJECT_DIR / "README.md"]
    files += list(PROJECT_DIR.glob("requirements*.txt"))
    files += [path for path in PROJECT_DIR.glob("DAY*.md") if "Draft" not in path.name]
    files += list(PROJECT_DIR.glob("DAY*.ipynb"))
    files += list((PROJECT_DIR / "src").glob("*.py"))
    files += [PROJECT_DIR / "data/README.md"]
    files += [path for path in (PROJECT_DIR / "results").rglob("*") if path.is_file()]
    files += list((PROJECT_DIR / "output/pdf").glob("DS-MINI-Design-*.pdf"))
    files = sorted(set(files))
    assert all(path.is_file() for path in files)
    assert not any(path.suffix == ".mat" for path in files)
    output_dir = PROJECT_DIR / "output/repository"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{REPOSITORY_NAME}.zip"
    manifest = {
        "repository_name": REPOSITORY_NAME,
        "files": [
            {
                "path": str(path.relative_to(PROJECT_DIR)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in files
        ],
        "contains_raw_mat": False,
        "public_repository_created": True,
        "repository_url": "https://github.com/miiiingyuuu/battery-cycle-life-prediction",
    }
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, f"{REPOSITORY_NAME}/{path.relative_to(PROJECT_DIR)}")
        archive.writestr(
            f"{REPOSITORY_NAME}/submission_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
    (output_dir / "package_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "file_count": len(files) + 1,
                "bytes": output.stat().st_size,
                "raw_mat": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return output


if __name__ == "__main__":
    package()
