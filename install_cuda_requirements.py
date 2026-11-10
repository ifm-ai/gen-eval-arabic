"""Install cuda_requirements.txt for the CUDA version reported by nvidia-smi.

PyTorch publishes wheels for a few CUDA versions. The script installs the
newest wheel at or below the driver, because a newer wheel will not load.
torch==2.13.0 is published for CUDA 12.6, 12.9, 13.0, and 13.2, so a driver
that reports 13.1 installs cu130 from https://download.pytorch.org/whl/cu130.
Toolkit packages follow that wheel: cu130 needs cuda-toolkit==13.0.3, which
is inside >13.0,<13.2. nvidia-cuda-nvdisasm stays at >=13.3,<14 because
CUTLASS 4.6.2 requires it and no 13.0 or 13.1 build is published.
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
# CUDA builds published for the torch pin in cuda_requirements.txt.
# Update this when that pin changes.
PYTORCH_CUDA_TAGS = ((12, 6), (12, 9), (13, 0), (13, 2))


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


def select_pytorch_cuda(driver):
    """Newest published PyTorch CUDA build that this driver can run."""
    compatible = [tag for tag in PYTORCH_CUDA_TAGS if tag <= driver]
    if not compatible:
        choices = ", ".join(f"{major}.{minor}" for major, minor in PYTORCH_CUDA_TAGS)
        sys.exit(
            f"CUDA {driver[0]}.{driver[1]} is older than the PyTorch wheels "
            f"pinned in {REQUIREMENTS.name} ({choices})."
        )
    return compatible[-1]


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

    driver = get_driver_cuda(args.cuda)
    if driver[0] not in (12, 13):
        sys.exit(
            f"CUDA {driver[0]}.{driver[1]} is outside the CUDA 12 and CUDA 13 packages in {REQUIREMENTS.name}. "
            "Install the CUDA requirements manually."
        )
    wheel = select_pytorch_cuda(driver)
    cuda, upper, cu_tag, resolved = resolve_packages(*wheel)
    print(f"Driver CUDA version: {driver[0]}.{driver[1]}")
    if wheel != driver:
        print(
            f"No PyTorch wheel is published for CUDA {driver[0]}.{driver[1]}. "
            f"Using CUDA {wheel[0]}.{wheel[1]} ({cu_tag}), the newest wheel this driver can run."
        )
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
