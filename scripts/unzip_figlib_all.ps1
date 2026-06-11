$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$srcDir = 'D:\detr_Q3\data\figlib'
$outDir = 'D:\detr_Q3\data\figlib_raw'
$logDir = 'D:\detr_Q3\idea-stage'
$logCsv = Join-Path $logDir 'figlib_unzip_log.csv'
$transcript = Join-Path $logDir 'figlib_unzip_transcript.txt'
$recordMd = Join-Path $logDir 'figlib_unzip_record.md'

New-Item -ItemType Directory -Force -Path $outDir, $logDir | Out-Null
if (-not (Test-Path $logCsv)) {
    'name,status,start_time,end_time,attempts,error' | Out-File -FilePath $logCsv -Encoding utf8
}

$archives = Get-ChildItem -Path $srcDir -Filter '*.tgz' | Sort-Object Name
$startAll = Get-Date
Start-Transcript -Path $transcript -Append | Out-Null
Write-Output "[start] $($startAll.ToString('yyyy-MM-dd HH:mm:ss')) total=$($archives.Count)"

foreach ($arc in $archives) {
    $name = $arc.Name
    $seq = [System.IO.Path]::GetFileNameWithoutExtension($name)
    $destDir = Join-Path $outDir $seq
    $start = Get-Date
    $attempts = 0

    if (Test-Path $destDir) {
        $hasFiles = Get-ChildItem -Path $destDir -File -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.Extension -ieq '.jpg' } |
            Select-Object -First 1
        if ($hasFiles) {
            "{0},exists,{1},{2},0," -f $name, $start.ToString('yyyy-MM-ddTHH:mm:ss'), $start.ToString('yyyy-MM-ddTHH:mm:ss') | Add-Content -Path $logCsv
            Write-Output "${name}: exists, skip"
            continue
        }
    }

    try {
        Remove-Item -Recurse -Force -Path $destDir -ErrorAction SilentlyContinue
        & tar -xzf $arc.FullName -C $outDir
        if ($LASTEXITCODE -ne 0) {
            throw "tar exit code=$LASTEXITCODE"
        }
        $hasAny = Get-ChildItem -Path $destDir -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $hasAny) {
            throw 'empty_extraction'
        }
        $attempts = 1
        "{0},success,{1},{2},{3}," -f $name, $start.ToString('yyyy-MM-ddTHH:mm:ss'), (Get-Date).ToString('yyyy-MM-ddTHH:mm:ss'), $attempts | Add-Content -Path $logCsv
        Write-Output "${name}: success"
    }
    catch {
        $attempts += 1
        $err = $_.Exception.Message
        if (Test-Path $destDir) {
            Remove-Item -Recurse -Force $destDir -ErrorAction SilentlyContinue
        }
        "{0},failed,{1},{2},{3},{4}" -f $name, $start.ToString('yyyy-MM-ddTHH:mm:ss'), (Get-Date).ToString('yyyy-MM-ddTHH:mm:ss'), $attempts, ($err -replace ',', ';') | Add-Content -Path $logCsv
        Write-Output "${name}: failed - $err"
    }
}

$endAll = Get-Date
$ok = (Import-Csv $logCsv | Where-Object { $_.status -in @('success', 'exists') }).Count
$fail = (Import-Csv $logCsv | Where-Object { $_.status -eq 'failed' }).Count
$summary = @"
# FIgLib 解压记录

- 开始时间: $($startAll.ToString('yyyy-MM-dd HH:mm:ss'))
- 结束时间: $($endAll.ToString('yyyy-MM-dd HH:mm:ss'))
- 耗时: $([int]($endAll - $startAll).TotalMinutes) 分钟
- 压缩源目录: $srcDir
- 目标目录: $outDir
- 总计: $($archives.Count)
- 成功(含已存在): $ok
- 失败: $fail
- 日志: $logCsv
- 实时日志: $transcript
"@
$summary | Set-Content -Path $recordMd -Encoding utf8
Write-Output "[end] $($endAll.ToString('yyyy-MM-dd HH:mm:ss')) ok=$ok fail=$fail"
Stop-Transcript | Out-Null