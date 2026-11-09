"""Install cuda_requirements.txt for the CUDA version reported by nvidia-smi.

A driver that reports CUDA 13.0 installs toolkit packages as >13.0,<13.2
and installs PyTorch from https://download.pytorch.org/whl/cu130.
nvidia-cuda-nvdisasm stays at >=13.3,<14 because CUTLASS 4.6.2 requires it
and no 13.0 or 13.1 build is published.
"""

import re
import sys
import subprocess
import tempfile
from pathlib import Path
from argparse import ArgumentParser

ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "cuda_requirements.txt"
DEFAULT_CUDA = (13, 0)

def parse_cuda(value):
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)", value.strip())
    if not match:
        sys.exit("--cuda must look like 13.0")
    return int(match.group(1)), int(match.group(2))


def cuda_from_nvidia_smi():
    try:
        output = subprocess.check_output(
            ["nvidia-smi"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError as exc:
        sys.exit(f"nvidia-smi failed:\n{exc.output}")
    match = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", output)
    if not match:
        sys.exit("Could not read 'CUDA Version' from nvidia-smi.")
    return int(match.group(1)), int(match.group(2))


def get_driver_cuda(override):
    if override:
        return parse_cuda(override)
    detected = cuda_from_nvidia_smi()
    if detected:
        return detected
    major, minor = DEFAULT_CUDA
    print(
        "nvidia-smi was not found. Installing the CUDA "
        f"{major}.{minor} packages from {REQUIREMENTS.name}. "
        "Pass --cuda to match a different driver, for example --cuda 12.8."
    )
    return DEFAULT_CUDA


def resolve_packages(major, minor):
    if major not in (12, 13):
        sys.exit(
            f"CUDA {major}.{minor} is outside the CUDA 12 and CUDA 13 packages in {REQUIREMENTS.name}. "
            "Install the CUDA requirements manually."
        )
    cuda = f"{major}.{minor}"
    upper = f"{major}.{minor + 2}"
    cu_tag = f"cu{major}{minor}"
    resolved = []
    for raw in REQUIREMENTS.read_text().splitlines():
        code, _, comment = raw.partition("#")
        code = code.strip()
        marker = comment.strip()
        if not code or "==" not in code:
            continue
        name, spec = code.split("==", 1)
        if marker == "cu12-only" and major != 12:
            continue
        if marker == "cu13-only" and major != 13:
            continue
        if marker == "cu-major" and major == 12:
            name = name.replace("-cu13", "-cu12")
        if name == "nvidia-cuda-nvdisasm":
            # CUTLASS 4.6.2 requires >=13.3,<14, and PyPI has no 13.0 or 13.1 build.
            resolved.append(f"{name}>=13.3,<14")
        elif marker == "toolkit":
            resolved.append(f"{name}>{cuda},<{upper}")
        elif marker == "torch":
            resolved.append(f"{name}=={spec.split('+', 1)[0]}+{cu_tag}")
        else:
            resolved.append(f"{name}=={spec}")
    return cuda, upper, cu_tag, resolved


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda",
        help="CUDA version to install for, for example 13.0. Default: nvidia-smi, or 13.0 when it is not available.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the package list and do not install."
    )
    args = parser.parse_args()

    if not REQUIREMENTS.is_file():
        sys.exit(f"Could not find {REQUIREMENTS}.")

    major, minor = get_driver_cuda(args.cuda)
    cuda, upper, cu_tag, resolved = resolve_packages(major, minor)
    print(f"Driver CUDA version: {cuda}")
    print(f"Toolkit packages: >{cuda},<{upper} (nvidia-cuda-nvdisasm>=13.3,<14)")
    print(f"PyTorch index: {cu_tag}")
    print()
    print("\n".join(resolved))

    if args.dry_run:
        return

    if sys.version_info < (3, 12) or sys.version_info >= (3, 15):
        sys.exit(
            f"Python {sys.version_info.major}.{sys.version_info.minor} cannot install these packages. "
            "Activate a virtual environment with Python >= 3.12 and < 3.15, then run this again."
        )

    generated = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", prefix="cuda_requirements-", suffix=".txt", delete=False
        ) as handle:
            handle.write("\n".join(resolved) + "\n")
            generated = handle.name
        print()
        print(f"Installing CUDA requirements from {generated}")
        install_command = [
            "uv", "pip", "install",
            "--extra-index-url", f"https://download.pytorch.org/whl/{cu_tag}",
            "--index-strategy", "unsafe-best-match",
            "-r", generated,
        ]
        try:
            subprocess.check_call(install_command)
        except FileNotFoundError:
            sys.exit(
                "uv was not found. Run `pip install uv --upgrade` from the installation steps, then try again."
            )
        except subprocess.CalledProcessError as exc:
            sys.exit(exc.returncode)
    finally:
        if generated is not None:
            Path(generated).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
