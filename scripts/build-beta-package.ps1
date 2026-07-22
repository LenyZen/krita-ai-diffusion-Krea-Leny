# Build a beta distributable zip for Krita AI Diffusion (ComfyforKrita fork).
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\build-beta-package.ps1
#   powershell -File scripts\build-beta-package.ps1 -VersionLabel "1.52.1-beta.3"

param(
    [string]$VersionLabel = "1.52.1-beta.3",
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"

$stamp = Get-Date -Format "yyyyMMdd"
$distRoot = Join-Path $Root "dist"
$stageName = "krita-ai-comfyfork-beta-$VersionLabel"
$stage = Join-Path $distRoot $stageName
$zipPath = Join-Path $distRoot "$stageName-$stamp.zip"

$pluginSrc = Join-Path $Root "ai_diffusion"
$desktopSrc = Join-Path $Root "ai_diffusion.desktop"
$toolingSrc = Join-Path $Root "krita-ai-krea2-describe\tooling-nodes-api.py"
$installSrc = Join-Path $Root "INSTALL-BETA.txt"
$changelogSrc = Join-Path $Root "CHANGELOG-BETA.txt"
$legacyInstallSrc = Join-Path $Root "INSTALL-KREA2-DESCRIBE.txt"

foreach ($p in @($pluginSrc, $desktopSrc, $toolingSrc, $installSrc, $changelogSrc)) {
    if (-not (Test-Path $p)) { throw "Missing required path: $p" }
}

Write-Host "Staging -> $stage"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path $stage | Out-Null

# Copy plugin tree, skipping caches and junk
$pluginDest = Join-Path $stage "ai_diffusion"
New-Item -ItemType Directory -Force -Path $pluginDest | Out-Null

$excludeDirs = @("__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache")
$excludeFiles = @("*.pyc", "*.pyo", "*.pyd", ".DS_Store", "Thumbs.db")

Get-ChildItem -Path $pluginSrc -Recurse -Force | ForEach-Object {
    $rel = $_.FullName.Substring($pluginSrc.Length).TrimStart("\", "/")
    $parts = $rel -split "[\\/]"
    foreach ($d in $excludeDirs) {
        if ($parts -contains $d) { return }
    }
    if (-not $_.PSIsContainer) {
        foreach ($pat in $excludeFiles) {
            if ($_.Name -like $pat) { return }
        }
        $dest = Join-Path $pluginDest $rel
        $destDir = Split-Path $dest -Parent
        if (-not (Test-Path $destDir)) {
            New-Item -ItemType Directory -Force -Path $destDir | Out-Null
        }
        Copy-Item -LiteralPath $_.FullName -Destination $dest -Force
    }
}

Copy-Item -LiteralPath $desktopSrc -Destination (Join-Path $stage "ai_diffusion.desktop") -Force
Copy-Item -LiteralPath $toolingSrc -Destination (Join-Path $stage "tooling-nodes-api.py") -Force
Copy-Item -LiteralPath $installSrc -Destination (Join-Path $stage "INSTALL-BETA.txt") -Force
Copy-Item -LiteralPath $changelogSrc -Destination (Join-Path $stage "CHANGELOG-BETA.txt") -Force
if (Test-Path $legacyInstallSrc) {
    Copy-Item -LiteralPath $legacyInstallSrc -Destination (Join-Path $stage "INSTALL-KREA2-DESCRIBE.txt") -Force
}

# Sanity checks
$websockets = Join-Path $pluginDest "websockets"
$initPy = Join-Path $pluginDest "__init__.py"
if (-not (Test-Path $websockets)) {
    throw "Packaged plugin is missing vendored websockets/ - do not ship a bare source tree."
}
$verLine = Select-String -Path $initPy -Pattern '__version__\s*=' | Select-Object -First 1
Write-Host "Packaged $($verLine.Line.Trim())"

if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
Write-Host "Compressing -> $zipPath"
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zipPath -CompressionLevel Optimal

$zipItem = Get-Item $zipPath
$fileCount = (Get-ChildItem -Path $stage -Recurse -File).Count
Write-Host ""
Write-Host "Done."
Write-Host "  Files staged : $fileCount"
Write-Host "  Zip size     : $([math]::Round($zipItem.Length / 1MB, 2)) MB"
Write-Host "  Zip path     : $($zipItem.FullName)"
Write-Host ""
Write-Host "Testers: Tools > Scripts > Import Python Plugin from File... then follow INSTALL-BETA.txt"
