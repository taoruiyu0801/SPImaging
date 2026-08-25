"""Publisher-signature hooks used after hash verification.

Unsigned beta manifests explicitly use ``kind=none``.  Production manifests
can require Authenticode on PE assets or a detached CMS signature on archives.
The latter is verified through the Windows .NET cryptography implementation so
the frozen launcher does not need a third-party cryptography package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile
from typing import Callable, Protocol

from .errors import SignatureError
from .manifest import ReleaseManifest, SignatureRequirement


# An unsigned public beta is a deliberately separate launcher trust mode. A
# valid Authenticode signature always upgrades the frozen launcher to the
# publisher-pinned mode below; a broken/invalid signature never falls back.
ALLOW_UNSIGNED_BETA_BUILD = True


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
    environment = os.environ.copy()
    # A launcher started from PowerShell 7 can inherit module paths that are
    # incompatible with Windows PowerShell 5.1. Give the system verifier only
    # its native module locations.
    system_root = Path(environment.get("SystemRoot", r"C:\Windows"))
    program_files = Path(environment.get("ProgramFiles", r"C:\Program Files"))
    profile = Path(environment.get("USERPROFILE", str(Path.cwd())))
    windows_module_path = os.pathsep.join(
        (
            str(profile / "Documents" / "WindowsPowerShell" / "Modules"),
            str(program_files / "WindowsPowerShell" / "Modules"),
            str(system_root / "System32" / "WindowsPowerShell" / "v1.0" / "Modules"),
        )
    )
    environment["PSModulePath"] = windows_module_path
    environment["SPIMAGING_WINDOWS_PSMODULEPATH"] = windows_module_path
    for index, argument in enumerate(arguments):
        environment[f"SPIMAGING_SIGNATURE_ARG_{index}"] = str(argument)
    try:
        completed = subprocess.run(
            [_powershell(), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            creationflags=creationflags,
            env=environment,
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
                "$env:PSModulePath=$env:SPIMAGING_WINDOWS_PSMODULEPATH;"
                "Import-Module Microsoft.PowerShell.Security;"
                "$s=Get-AuthenticodeSignature -LiteralPath $env:SPIMAGING_SIGNATURE_ARG_0;"
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
                "$content=[IO.File]::ReadAllBytes($env:SPIMAGING_SIGNATURE_ARG_0);"
                "$sig=[IO.File]::ReadAllBytes($env:SPIMAGING_SIGNATURE_ARG_1);"
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


@dataclass(frozen=True)
class AuthenticodeIdentity:
    status: str
    signer_thumbprint: str | None


def inspect_authenticode(path: Path) -> AuthenticodeIdentity:
    """Inspect the launcher itself without trusting release-manifest fields."""

    script = (
        "$ErrorActionPreference='Stop';"
        "$env:PSModulePath=$env:SPIMAGING_WINDOWS_PSMODULEPATH;"
        "Import-Module Microsoft.PowerShell.Security;"
        "$s=Get-AuthenticodeSignature -LiteralPath $env:SPIMAGING_SIGNATURE_ARG_0;"
        "$thumb=$null;if($s.SignerCertificate){$thumb=$s.SignerCertificate.Thumbprint.ToUpperInvariant()};"
        "[pscustomobject]@{status=$s.Status.ToString();thumbprint=$thumb}|ConvertTo-Json -Compress"
    )
    raw = _run_powershell(script, path.resolve(strict=True))
    try:
        value = json.loads(raw)
        status = value["status"]
        thumbprint = value.get("thumbprint")
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise SignatureError("could not parse launcher Authenticode identity") from error
    if not isinstance(status, str):
        raise SignatureError("launcher Authenticode status is invalid")
    normalized = None
    if thumbprint is not None:
        if not isinstance(thumbprint, str):
            raise SignatureError("launcher signer thumbprint is invalid")
        normalized = thumbprint.replace(" ", "").upper()
    if status == "Valid":
        if normalized is None:
            raise SignatureError("valid launcher signature has no signer certificate")
        return AuthenticodeIdentity(status, normalized)
    if status == "NotSigned":
        return AuthenticodeIdentity(status, None)
    raise SignatureError(f"launcher Authenticode signature is not valid: {status}")


def verify_detached_cms_bytes(content: bytes, signature: bytes, expected_thumbprint: str) -> None:
    """Verify detached CMS bytes using the existing Windows verifier."""

    if not signature:
        raise SignatureError("release manifest detached signature is empty")
    with tempfile.TemporaryDirectory(prefix="spimaging-manifest-") as temporary:
        root = Path(temporary)
        artifact = root / "release-manifest.json"
        signature_path = root / "release-manifest.json.p7s"
        artifact.write_bytes(content)
        signature_path.write_bytes(signature)
        WindowsSignatureVerifier().verify(
            artifact,
            SignatureRequirement(
                "cms-detached",
                True,
                signer_thumbprint=expected_thumbprint,
                file_name=signature_path.name,
            ),
            signature_path,
        )


@dataclass(frozen=True)
class ManifestTrustPolicy:
    """Launcher-owned trust root for release manifests and their assets."""

    mode: str
    signer_thumbprint: str | None = None
    detached_verifier: Callable[[bytes, bytes, str], None] = field(
        default=verify_detached_cms_bytes,
        repr=False,
        compare=False,
    )

    @classmethod
    def unsigned_beta(cls) -> "ManifestTrustPolicy":
        return cls("unsigned-beta")

    @classmethod
    def signed(
        cls,
        signer_thumbprint: str,
        *,
        detached_verifier: Callable[[bytes, bytes, str], None] = verify_detached_cms_bytes,
    ) -> "ManifestTrustPolicy":
        normalized = signer_thumbprint.replace(" ", "").upper()
        if re.fullmatch(r"[0-9A-F]{40}|[0-9A-F]{64}", normalized) is None:
            raise SignatureError("signed manifest policy requires a SHA-1 or SHA-256 publisher thumbprint")
        return cls("publisher-signed", normalized, detached_verifier)

    @classmethod
    def current(cls) -> "ManifestTrustPolicy":
        # Source checkouts are development/unsigned-beta mode. Frozen Windows
        # launchers derive the formal trust root from their own signed PE file.
        if not getattr(sys, "frozen", False):
            return cls.unsigned_beta()
        identity = inspect_authenticode(Path(sys.executable))
        if identity.status == "Valid" and identity.signer_thumbprint:
            return cls.signed(identity.signer_thumbprint)
        if identity.status == "NotSigned" and ALLOW_UNSIGNED_BETA_BUILD:
            return cls.unsigned_beta()
        raise SignatureError("this launcher build has no permitted release-manifest trust mode")

    @property
    def signature_required(self) -> bool:
        return self.mode == "publisher-signed"

    def verify_manifest(
        self,
        raw: bytes,
        manifest: ReleaseManifest,
        signature: bytes | None,
    ) -> None:
        try:
            reparsed = ReleaseManifest.from_json(raw)
        except Exception as error:
            raise SignatureError(f"release manifest bytes are invalid: {error}") from error
        if reparsed != manifest:
            raise SignatureError("release manifest changed between parsing and trust verification")
        if self.mode == "unsigned-beta":
            if not manifest.unsigned_beta or manifest.channel != "beta":
                raise SignatureError("unsigned-beta launcher refuses signed/stable release manifests")
            if any(asset.signature.required or asset.signature.kind != "none" for asset in manifest.assets):
                raise SignatureError("unsigned-beta manifest must contain only explicitly unsigned assets")
            if signature is not None:
                raise SignatureError("unsigned-beta trust mode refuses ambiguous detached manifest signatures")
            return
        if self.mode != "publisher-signed" or not self.signer_thumbprint:
            raise SignatureError("release-manifest trust policy is invalid")
        if manifest.unsigned_beta:
            raise SignatureError("signed launcher refuses an unsigned-beta release manifest")
        if signature is None:
            raise SignatureError("signed release manifest is missing its detached CMS signature")
        for asset in manifest.assets:
            requirement = asset.signature
            if (
                not requirement.required
                or requirement.kind != "cms-detached"
                or requirement.signer_thumbprint != self.signer_thumbprint
            ):
                raise SignatureError(
                    f"asset {asset.asset_id} is not pinned to the launcher's Authenticode publisher"
                )
        self.detached_verifier(raw, signature, self.signer_thumbprint)
