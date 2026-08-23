"""Publisher-signature hooks used after hash verification.

Unsigned beta manifests explicitly use ``kind=none``.  Production manifests
can require Authenticode on PE assets or a detached CMS signature on archives.
The latter is verified through the Windows .NET cryptography implementation so
the frozen launcher does not need a third-party cryptography package.
"""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
from typing import Protocol

from .errors import SignatureError
from .manifest import SignatureRequirement


class SignatureVerifier(Protocol):
    def verify(
        self,
        artifact: Path,
        requirement: SignatureRequirement,
        signature_file: Path | None = None,
    ) -> None: ...


def _powershell() -> str:
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    candidate = system_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    return str(candidate if candidate.exists() else "powershell.exe")


def _run_powershell(script: str, *arguments: Path | str) -> str:
    if os.name != "nt":
        raise SignatureError("publisher signature verification requires Windows")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            [_powershell(), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script, *map(str, arguments)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            creationflags=creationflags,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SignatureError(f"could not invoke Windows signature verifier: {error}") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "signature check failed").strip()
        raise SignatureError(detail)
    return completed.stdout.strip()


class WindowsSignatureVerifier:
    """Verify a required signature and pin the exact publisher certificate."""

    def verify(
        self,
        artifact: Path,
        requirement: SignatureRequirement,
        signature_file: Path | None = None,
    ) -> None:
        if not requirement.required:
            return
        artifact = artifact.resolve(strict=True)
        expected = requirement.signer_thumbprint or ""
        if requirement.kind == "authenticode":
            script = (
                "$ErrorActionPreference='Stop';"
                "$s=Get-AuthenticodeSignature -LiteralPath $args[0];"
                "if($s.Status -ne 'Valid'){throw ('Authenticode status: '+$s.StatusMessage)};"
                "$s.SignerCertificate.Thumbprint.ToUpperInvariant()"
            )
            actual = _run_powershell(script, artifact).replace(" ", "").upper()
        elif requirement.kind == "cms-detached":
            if signature_file is None:
                raise SignatureError("detached CMS signature file is missing")
            signature_file = signature_file.resolve(strict=True)
            script = (
                "$ErrorActionPreference='Stop';"
                "Add-Type -AssemblyName System.Security;"
                "$content=[IO.File]::ReadAllBytes($args[0]);"
                "$sig=[IO.File]::ReadAllBytes($args[1]);"
                "$info=[Security.Cryptography.Pkcs.ContentInfo]::new([byte[]]$content);"
                "$cms=[Security.Cryptography.Pkcs.SignedCms]::new($info,$true);"
                "$cms.Decode($sig);$cms.CheckSignature($false);"
                "if($cms.SignerInfos.Count -ne 1){throw 'Expected exactly one CMS signer'};"
                "$cms.SignerInfos[0].Certificate.Thumbprint.ToUpperInvariant()"
            )
            actual = _run_powershell(script, artifact, signature_file).replace(" ", "").upper()
        else:
            raise SignatureError(f"unsupported required signature kind: {requirement.kind}")
        if actual != expected:
            raise SignatureError(f"publisher certificate thumbprint mismatch: expected {expected}, got {actual}")
