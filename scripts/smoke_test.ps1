param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [int]$Attempts = 12,
    [int]$RetrySeconds = 5
)

$ErrorActionPreference = "Stop"
$health = $null
$lastError = $null

for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri "$BaseUrl/health" -TimeoutSec 15
        if ($health.model_loaded) {
            break
        }
        $lastError = "Service is not ready: $($health | ConvertTo-Json -Compress)"
    }
    catch {
        $lastError = $_.Exception.Message
    }

    if ($attempt -lt $Attempts) {
        Start-Sleep -Seconds $RetrySeconds
    }
}

if (-not $health -or -not $health.model_loaded) {
    throw "Service did not become ready after $Attempts attempts. Last error: $lastError"
}

$smokeImage = Join-Path ([System.IO.Path]::GetTempPath()) "cats-dogs-smoke-$PID.ppm"
try {
    $header = [System.Text.Encoding]::ASCII.GetBytes("P6`n1 1`n255`n")
    $pixel = [byte[]](128, 128, 128)
    [System.IO.File]::WriteAllBytes($smokeImage, $header + $pixel)

    $prediction = Invoke-RestMethod -Uri "$BaseUrl/predict" -Method Post -Form @{
        file = Get-Item -LiteralPath $smokeImage
    } -TimeoutSec 30

    if ($prediction.label -notin @("cat", "dog")) {
        throw "Unexpected prediction label: $($prediction.label)"
    }
    if ($null -eq $prediction.probabilities.cat -or $null -eq $prediction.probabilities.dog) {
        throw "Prediction response is missing cat/dog probabilities."
    }

    [pscustomobject]@{
        health = $health
        prediction = $prediction
    } | ConvertTo-Json -Depth 5
}
finally {
    Remove-Item -LiteralPath $smokeImage -Force -ErrorAction SilentlyContinue
}
