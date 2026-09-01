[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8080,
    [switch]$SkipSync,
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot
$env:UV_CACHE_DIR = Join-Path $projectRoot ".uv-cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot ".uv-python"

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host "[AbstractTikTok] $Label" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($null -eq $uvCommand) {
    throw "uv is required. Install it from https://docs.astral.sh/uv/ and run this script again."
}

if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
    Write-Host "[AbstractTikTok] Created .env from .env.example. Add API keys when AI features are needed." -ForegroundColor Yellow
}

if (-not $SkipSync) {
    Invoke-CheckedCommand "Synchronizing locked dependencies" { & $uvCommand.Source sync --frozen }
}

Invoke-CheckedCommand "Applying analysis database migrations" { & $uvCommand.Source run alembic upgrade head }

$url = "http://127.0.0.1:$Port/"
$dashboardUrl = "http://127.0.0.1:$Port/dashboard"
$browserJob = $null
if (-not $NoBrowser) {
    $browserJob = Start-Job -ScriptBlock {
        param($TargetUrl)
        for ($attempt = 0; $attempt -lt 60; $attempt++) {
            try {
                Invoke-WebRequest -Uri $TargetUrl -UseBasicParsing -TimeoutSec 1 | Out-Null
                Start-Process $TargetUrl
                return
            }
            catch {
                Start-Sleep -Milliseconds 500
            }
        }
    } -ArgumentList $url
}

Write-Host "[AbstractTikTok] Crawler UI: $url" -ForegroundColor Green
Write-Host "[AbstractTikTok] AI Dashboard: $dashboardUrl" -ForegroundColor Green
Write-Host "[AbstractTikTok] Press Ctrl+C to stop." -ForegroundColor DarkGray

try {
    & $uvCommand.Source run uvicorn api.main:app --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) {
        throw "Web server exited with code $LASTEXITCODE"
    }
}
finally {
    if ($null -ne $browserJob) {
        Stop-Job -Job $browserJob -ErrorAction SilentlyContinue
        Remove-Job -Job $browserJob -Force -ErrorAction SilentlyContinue
    }
}
