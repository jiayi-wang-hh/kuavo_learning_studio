param(
    [string]$Ref = "4af2b62",
    [switch]$Install
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Source = Join-Path $Root "source"

if (Test-Path $Source) {
    throw "GR00T source already exists at $Source. Remove it explicitly before bootstrapping again."
}

git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/Isaac-GR00T.git $Source
git -C $Source checkout $Ref

if ($Install) {
    uv sync --project $Source
}

Write-Host "Isaac-GR00T N1.5 source ready at $Source (ref $Ref)"
