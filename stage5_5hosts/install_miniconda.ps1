$ErrorActionPreference = "Stop"

$target = Join-Path $env:USERPROFILE "miniconda3"
$activate = Join-Path $target "Scripts\activate.bat"

if (Test-Path $activate) {
    Write-Host "[ok] Miniconda already exists: $activate"
    exit 0
}

if (Test-Path $target) {
    throw "Target directory exists but activate.bat is missing: $target. Fix or remove it manually, then rerun."
}

$urls = @(
    "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe",
    "https://mirrors.ustc.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe",
    "https://mirrors.bfsu.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe"
)
$installer = Join-Path $env:TEMP "Miniconda3-latest-Windows-x86_64.exe"

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$downloaded = $false
foreach ($url in $urls) {
    try {
        Write-Host "[info] Downloading Miniconda from mirror: $url"
        Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing
        $downloaded = $true
        break
    } catch {
        Write-Host "[warn] Mirror failed: $url"
        Write-Host "[warn] $($_.Exception.Message)"
    }
}

if (-not $downloaded) {
    throw "All Miniconda mirrors failed. Check network or download Miniconda manually into $installer."
}

Write-Host "[info] Installing Miniconda into $target"
$installerArgs = @(
    "/InstallationType=JustMe",
    "/AddToPath=0",
    "/RegisterPython=0",
    "/S",
    "/D=$target"
)
$process = Start-Process -FilePath $installer -ArgumentList $installerArgs -Wait -PassThru
if ($process.ExitCode -ne 0) {
    throw "Miniconda installer failed with exit code $($process.ExitCode)"
}

if (!(Test-Path $activate)) {
    throw "Miniconda install finished, but activate.bat was not found: $activate"
}

Write-Host "[ok] Miniconda installed: $activate"
