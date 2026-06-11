$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$baseUrl = 'https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar/'
$indexUrl = 'https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar/index.html'
$outDir = 'D:\detr_Q3\data\figlib'
$recordDir = 'D:\detr_Q3\idea-stage'
$recordCsv = Join-Path $recordDir 'figlib_download_log.csv'
$recordMd = Join-Path $recordDir 'figlib_download_record.md'
$transcript = Join-Path $recordDir 'figlib_download_transcript.txt'
$retryCount = 5
$timeoutSeconds = 900

New-Item -ItemType Directory -Force -Path $outDir, $recordDir | Out-Null

function Test-GzipMagic([string]$Path) {
    if (-not (Test-Path -Path $Path -PathType Leaf)) { return $false }
    if ((Get-Item $Path).Length -lt 4096) { return $false }
    $fs = [System.IO.File]::OpenRead($Path)
    try {
        $bytes = New-Object byte[] 2
        $n = $fs.Read($bytes, 0, 2)
        if ($n -lt 2) { return $false }
        return ('{0:X2}{1:X2}' -f $bytes[0], $bytes[1]) -eq '1F8B'
    }
    finally {
        $fs.Dispose()
    }
}

$index = Invoke-WebRequest -Uri $indexUrl -UseBasicParsing
$links = @($index.Links | Where-Object { $_.href -match '\.tgz$' } | Select-Object -ExpandProperty href -Unique)
$links = @($links | Sort-Object)

if (-not (Test-Path -Path $recordCsv)) {
    "name,status,start_time,end_time,attempts,error`n" | Out-File -FilePath $recordCsv -Encoding utf8
}

Start-Transcript -Path $transcript -Append | Out-Null
$startAll = Get-Date
Write-Output "[start] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') total=$($links.Count)"

$success = 0
$skip = 0
$fail = 0

foreach ($name in $links) {
    $start = Get-Date
    $url = $baseUrl + $name
    $dest = Join-Path $outDir $name
    $status = 'failed'
    $err = ''
    $attempts = 0

    if (Test-Path -Path $dest -PathType Leaf) {
        if (Test-GzipMagic -Path $dest) {
            $success++
            $skip++
            "{0},exists,{1},{2},0," -f $name, $start.ToString('yyyy-MM-ddTHH:mm:ss'), $start.ToString('yyyy-MM-ddTHH:mm:ss') | Add-Content -Path $recordCsv
            continue
        }
        Remove-Item -Force -Path $dest -ErrorAction SilentlyContinue
    }

    for ($i = 1; $i -le $retryCount; $i++) {
        $attempts = $i
        try {
            & curl.exe -k -L --http1.1 --connect-timeout 20 --max-time $timeoutSeconds --retry 0 --retry-delay 2 --show-error --fail -C - -o $dest $url | Out-Null
            if ($LASTEXITCODE -ne 0) {
                throw "curl exit code=$LASTEXITCODE"
            }
            if (Test-GzipMagic -Path $dest) {
                $status = 'success'
                $success++
                break
            }
            throw "file_not_gzip_magic:$((Get-Item $dest).Length)"
        }
        catch {
            $err = $_.Exception.Message
            if ($i -lt $retryCount) {
                Remove-Item -Force -Path $dest -ErrorAction SilentlyContinue
                Start-Sleep -Seconds ($i * 2)
            }
        }
    }

    if ($status -eq 'success') {
        "{0},success,{1},{2},{3},{4}" -f $name, $start.ToString('yyyy-MM-ddTHH:mm:ss'), (Get-Date).ToString('yyyy-MM-ddTHH:mm:ss'), $attempts, ($err -replace ',', ';') | Add-Content -Path $recordCsv
        Write-Output "{0}: success in {1} attempts" -f $name, $attempts
    }
    else {
        $fail++
        if (Test-Path -Path $dest -PathType Leaf) { Remove-Item -Force -Path $dest }
        "{0},failed,{1},{2},{3},{4}" -f $name, $start.ToString('yyyy-MM-ddTHH:mm:ss'), (Get-Date).ToString('yyyy-MM-ddTHH:mm:ss'), $attempts, ($err -replace ',', ';') | Add-Content -Path $recordCsv
        Write-Output "{0}: failed after {1} attempts" -f $name, $attempts
    }
}

$endAll = Get-Date
$summary = @"
# FIgLib 下载记录

- 开始时间: $($startAll.ToString('yyyy-MM-dd HH:mm:ss'))
- 结束时间: $($endAll.ToString('yyyy-MM-dd HH:mm:ss'))
- 耗时: $([int]($endAll - $startAll).TotalMinutes) 分钟
- 数据源: https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar/index.html
- 目标目录: $outDir
- 总计: $($links.Count)
- 成功(含跳过): $success
- 新下载成功: $($success - $skip)
- 已存在复用: $skip
- 失败: $fail
- 日志: $recordCsv
- 实时日志: $transcript
"@
$summary | Set-Content -Path $recordMd -Encoding utf8

Write-Output "[end] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') success=$success skip=$skip fail=$fail"
Stop-Transcript | Out-Null