param(
  [Parameter(Mandatory = $true)]
  [string]$Server,
  [string]$RemoteRoot = "/data/detr_Q3",
  [int]$Port = 0
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$SshPortArgs = @()
$ScpPortArgs = @()
if ($Port -gt 0) {
  $SshPortArgs = @("-p", "$Port")
  $ScpPortArgs = @("-P", "$Port")
}

Write-Host "[info] local root: $Root"
Write-Host "[info] remote root: ${Server}:$RemoteRoot"
ssh @SshPortArgs $Server "mkdir -p '$RemoteRoot'"

$items = @(
  "scripts",
  "stage5_h20_server",
  "stage5_5hosts",
  "stage5_pv_v2",
  "data\dfire_local",
  "data\dfire",
  "data\D-Fire",
  "data\stage5_pv_v2",
  "external_data\BoWFireDataset",
  "external_data\DFS-FIRE-SMOKE-Dataset",
  "rtdetr-l.pt",
  "yolo26n.pt",
  "MANIFEST.md",
  "STAGE5_H20_COMMANDS.md"
)

foreach ($item in $items) {
  $src = Join-Path $Root $item
  if (Test-Path $src) {
    $remoteItem = $item.Replace("\", "/")
    $remoteParent = ""
    if ($remoteItem.Contains("/")) {
      $remoteParent = $remoteItem.Substring(0, $remoteItem.LastIndexOf("/"))
    }
    $remoteDestDir = $RemoteRoot
    if ($remoteParent.Length -gt 0) {
      $remoteDestDir = "$RemoteRoot/$remoteParent"
    }
    $target = "${Server}:$remoteDestDir/"
    Write-Host "[sync] $item"
    ssh @SshPortArgs $Server "mkdir -p '$remoteDestDir'"
    scp @ScpPortArgs -r $src $target
  } else {
    Write-Host "[skip] missing: $item"
  }
}

ssh @SshPortArgs $Server "cd '$RemoteRoot' && chmod +x stage5_h20_server/*.sh && (python3 stage5_h20_server/02_rewrite_paths_for_server.py --root '$RemoteRoot' --dry-run || true)"
Write-Host "[ok] sync finished. Next: ssh $Server `"cd '$RemoteRoot' && bash stage5_h20_server/00_bootstrap_h20_env.sh`""
