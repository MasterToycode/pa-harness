# Load API keys from DSH's credential store into THIS PowerShell session.
#
#   . .\scripts\load-keys.ps1                  # all known keys
#   . .\scripts\load-keys.ps1 -Only DEEPSEEK_API_KEY,QWEN_API_KEY
#
# Why this exists: `pah` reads keys from environment variables only (never from a
# config file), while the values on this machine live in DSH's credential store.
# Without a loader you retype the key in every new shell — and a key pasted into a
# terminal ends up in scrollback. This script moves it into the process environment
# and prints only variable names.
#
# It writes nothing to disk and prints no value. Dot-source it so the variables land
# in your current shell:
#
#   PS> . .\scripts\load-keys.ps1
#   PS> pah providers doctor --role llm
#
# For a permanent setup use `setx DEEPSEEK_API_KEY "..."` once instead (new shells
# pick it up automatically), or keep using this loader per session.

param(
    [string]$Credentials = "$env:USERPROFILE\.dsh\.credentials.yaml",
    [string[]]$Only = @()
)

if (-not (Test-Path $Credentials)) {
    Write-Error "credential store not found: $Credentials"
    return
}

# Matches `NAME: value` lines whose name looks like a credential. Deliberately does
# not match `maxTokens`-style settings, which are limits, not secrets.
$pattern = '^\s*([A-Z][A-Z0-9_]*(?:_API_KEY|_KEY|_TOKEN|_SECRET))\s*:\s*(\S+)\s*$'
$loaded = @()

foreach ($line in Get-Content $Credentials) {
    $match = [regex]::Match($line, $pattern)
    if (-not $match.Success) { continue }
    $name = $match.Groups[1].Value
    $value = $match.Groups[2].Value
    if ($Only.Count -gt 0 -and $Only -notcontains $name) { continue }
    Set-Item -Path "Env:$name" -Value $value
    $loaded += $name
}

if ($loaded.Count -eq 0) {
    Write-Warning "no credential matched in $Credentials (use -Only to name one explicitly)"
} else {
    Write-Output ("loaded into this session: " + ($loaded -join ", "))
}
