$ErrorActionPreference = "Stop"

function Run([string]$cmd) {
    Write-Host ">> $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { throw "Adim basarisiz: $cmd" }
}

Run "python train.py"
Run "docker build -t iris-api ."

$old = docker ps -aq -f "name=ci-api"
if ($old) { docker rm -f ci-api | Out-Null }
Run "docker run -d -p 8001:8000 --name ci-api iris-api"

try {
    $ok = $false
    for ($i = 0; $i -lt 20; $i++) {
        try {
            Invoke-RestMethod http://localhost:8001/health | Out-Null
            $ok = $true
            break
        } catch { Start-Sleep 1 }
    }
    if (-not $ok) { throw "API ayaga kalkmadi" }

    $body = '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
    $r = Invoke-RestMethod -Method Post -Uri http://localhost:8001/predict -ContentType "application/json" -Body $body
    if ($r.class_name -ne "setosa") { throw "Beklenmeyen tahmin: $($r.class_name)" }

    Write-Host "TUM ADIMLAR BASARILI" -ForegroundColor Green
}
finally {
    docker rm -f ci-api | Out-Null
}
