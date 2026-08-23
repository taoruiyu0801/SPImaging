[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Artifact,
    [ValidateSet("Authenticode", "DetachedCms")]
    [string]$Mode = "Authenticode",
    [string]$CertificateThumbprint,
    [string]$PfxPath,
    [string]$PfxPasswordEnvironmentVariable = "SPIMAGING_PFX_PASSWORD",
    [string]$TimestampUrl = "http://timestamp.digicert.com"
)

$ErrorActionPreference = "Stop"
$artifactPath = [IO.Path]::GetFullPath($Artifact)
if (-not (Test-Path -LiteralPath $artifactPath -PathType Leaf)) { throw "Artifact not found: $artifactPath" }
if (-not $CertificateThumbprint -and -not $PfxPath) {
    throw "Signing is conditional: provide an existing -CertificateThumbprint or -PfxPath. No certificate will be created."
}

if ($PfxPath) {
    $password = [Environment]::GetEnvironmentVariable($PfxPasswordEnvironmentVariable)
    if (-not $password) { throw "Missing PFX password environment variable: $PfxPasswordEnvironmentVariable" }
    $securePassword = ConvertTo-SecureString $password -AsPlainText -Force
    $certificate = New-Object Security.Cryptography.X509Certificates.X509Certificate2(
        [IO.Path]::GetFullPath($PfxPath), $securePassword,
        [Security.Cryptography.X509Certificates.X509KeyStorageFlags]::EphemeralKeySet
    )
} else {
    $certificate = Get-ChildItem Cert:\CurrentUser\My | Where-Object Thumbprint -eq $CertificateThumbprint | Select-Object -First 1
    if (-not $certificate) { throw "Signing certificate not found in CurrentUser/My: $CertificateThumbprint" }
}
if ($CertificateThumbprint) {
    $expectedThumbprint = ($CertificateThumbprint -replace '\s', '').ToUpperInvariant()
    $actualThumbprint = ($certificate.Thumbprint -replace '\s', '').ToUpperInvariant()
    if ($actualThumbprint -ne $expectedThumbprint) {
        throw "Signing certificate thumbprint mismatch: expected $expectedThumbprint, got $actualThumbprint"
    }
}

if ($Mode -eq "Authenticode") {
    $signature = Set-AuthenticodeSignature -LiteralPath $artifactPath -Certificate $certificate -HashAlgorithm SHA256 -TimestampServer $TimestampUrl
    if ($signature.Status -ne "Valid") { throw "Authenticode signing failed: $($signature.StatusMessage)" }
    Write-Host "Authenticode signed $artifactPath"
    exit 0
}

Add-Type -AssemblyName System.Security
$content = [IO.File]::ReadAllBytes($artifactPath)
$contentInfo = [Security.Cryptography.Pkcs.ContentInfo]::new([byte[]]$content)
$cms = [Security.Cryptography.Pkcs.SignedCms]::new($contentInfo, $true)
$signer = [Security.Cryptography.Pkcs.CmsSigner]::new($certificate)
$signer.DigestAlgorithm = [Security.Cryptography.Oid]::new("2.16.840.1.101.3.4.2.1")
$cms.ComputeSignature($signer)
$signaturePath = "$artifactPath.p7s"
[IO.File]::WriteAllBytes($signaturePath, $cms.Encode())
Write-Host "Detached CMS signed $artifactPath -> $signaturePath"
