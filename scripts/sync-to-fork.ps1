# Sync local ComfyforKrita plugin worktree into the GitHub fork clone.
# Does NOT push. Preserves git submodules (websockets, debugpy).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\sync-to-fork.ps1
#   powershell -File scripts\sync-to-fork.ps1 -Dest "D:\krita-ai-diffusion-Krea-Leny"
#   powershell -File scripts\sync-to-fork.ps1 -DryRun
#   powershell -File scripts\sync-to-fork.ps1 -Commit -Push
#
# After sync (without -Commit):
#   cd <Dest>
#   git add -A
#   git commit ...
#   git push

param(
    [string]$Source = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$Dest = "E:\krita-ai-diffusion-Krea-Leny",
    [switch]$DryRun,
    [switch]$Commit,
    [switch]$Push,
    [string]$CommitMessage = "Sync beta plugin changes from ComfyforKrita worktree."
)

$ErrorActionPreference = "Stop"

function Assert-Path([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing $Label : $Path"
    }
}

function Invoke-Robo([string[]]$ArgsList) {
    # robocopy exit codes 0-7 are success-ish; >=8 is failure
    & robocopy @ArgsList | Out-Null
    $code = $LASTEXITCODE
    if ($code -ge 8) {
        throw "robocopy failed (exit $code): $($ArgsList -join ' ')"
    }
    return $code
}

Assert-Path $Source "Source"
Assert-Path (Join-Path $Source "ai_diffusion") "Source plugin"
Assert-Path $Dest "Dest fork clone"
Assert-Path (Join-Path $Dest ".git") "Dest .git (clone the fork first)"

$pluginSrc = Join-Path $Source "ai_diffusion"
$pluginDst = Join-Path $Dest "ai_diffusion"

Write-Host "Source : $Source"
Write-Host "Dest   : $Dest"
if ($DryRun) { Write-Host "Mode   : DRY RUN (no files written)" }

# Submodules / caches must never be mirrored away
$excludeDirs = @(
    "__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "websockets", "debugpy"
)
$excludeFiles = @("*.pyc", "*.pyo", "*.pyd", ".DS_Store", "Thumbs.db")

$roboCommon = @("/NFL", "/NDL", "/NJH", "/NJS", "/nc", "/ns", "/np")
if ($DryRun) { $roboCommon += "/L" }

# /MIR keeps dest aligned with source, but skip submodule folders
$xd = @()
foreach ($d in $excludeDirs) { $xd += @("/XD", $d) }
$xf = @()
foreach ($f in $excludeFiles) { $xf += @("/XF", $f) }

Write-Host "Syncing ai_diffusion (preserving websockets/ + debugpy/ submodules)..."
Invoke-Robo (@($pluginSrc, $pluginDst, "/MIR") + $roboCommon + $xd + $xf) | Out-Null

# Root / docs / scripts extras
$copyMap = @(
    @{ From = "ai_diffusion.desktop"; To = "ai_diffusion.desktop"; Required = $true },
    @{ From = "INSTALL-BETA.txt"; To = "INSTALL-BETA.txt"; Required = $true },
    @{ From = "CHANGELOG-BETA.txt"; To = "CHANGELOG-BETA.txt"; Required = $true },
    @{ From = "INSTALL-KREA2-DESCRIBE.txt"; To = "INSTALL-KREA2-DESCRIBE.txt"; Required = $false },
    @{ From = "scripts\build-beta-package.ps1"; To = "scripts\build-beta-package.ps1"; Required = $true },
    @{ From = "scripts\sync-to-fork.ps1"; To = "scripts\sync-to-fork.ps1"; Required = $true },
    @{ From = "krita-ai-krea2-describe\tooling-nodes-api.py"; To = "tooling-nodes-api.py"; Required = $false },
    @{ From = "tooling-nodes-api.py"; To = "tooling-nodes-api.py"; Required = $false }
)

foreach ($item in $copyMap) {
    $from = Join-Path $Source $item.From
    $to = Join-Path $Dest $item.To
    if (-not (Test-Path -LiteralPath $from)) {
        if ($item.Required) { throw "Missing required file: $from" }
        continue
    }
    if ($DryRun) {
        Write-Host "Would copy $($item.From) -> $($item.To)"
        continue
    }
    $toDir = Split-Path $to -Parent
    if (-not (Test-Path $toDir)) {
        New-Item -ItemType Directory -Force -Path $toDir | Out-Null
    }
    Copy-Item -LiteralPath $from -Destination $to -Force
    Write-Host "Copied $($item.To)"
}

$docsSrc = Join-Path $Source "docs"
$docsDst = Join-Path $Dest "docs"
if (Test-Path $docsSrc) {
    Write-Host "Syncing docs\ ..."
    Invoke-Robo (@($docsSrc, $docsDst, "/E") + $roboCommon + @("/XD", "__pycache__") + $xf) | Out-Null
}

# Sanity: submodules still present
foreach ($sub in @("websockets", "debugpy")) {
    $p = Join-Path $pluginDst $sub
    if (-not (Test-Path $p)) {
        Write-Warning "Submodule folder missing after sync: $p - run: git -C `"$Dest`" submodule update --init ai_diffusion/$sub"
    }
}

if ($DryRun) {
    Write-Host "Dry run complete."
    exit 0
}

Write-Host ""
Write-Host "Sync complete."
Push-Location $Dest
try {
    git status -sb
    if ($Commit) {
        git add -A
        $pending = git status --porcelain
        if (-not $pending) {
            Write-Host "Nothing to commit."
        }
        else {
            git -c user.name="LenyZen" -c user.email="LenyZen@users.noreply.github.com" commit -m $CommitMessage
            Write-Host "Committed."
            if ($Push) {
                git push origin HEAD
                Write-Host "Pushed."
            }
            else {
                Write-Host "Not pushed. Run: git -C `"$Dest`" push"
            }
        }
    }
    else {
        Write-Host ""
        Write-Host "Next:"
        Write-Host "  git -C `"$Dest`" add -A"
        Write-Host "  git -C `"$Dest`" commit -m `"your message`""
        Write-Host "  git -C `"$Dest`" push"
        Write-Host "Or re-run with -Commit -Push"
    }
}
finally {
    Pop-Location
}
