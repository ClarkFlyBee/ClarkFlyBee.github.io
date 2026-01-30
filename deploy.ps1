# deploy.ps1
$ErrorActionPreference = "Stop"

Write-Host "Cleaning old build..." -ForegroundColor Green
Remove-Item -Recurse -Force public/ -ErrorAction SilentlyContinue

Write-Host "Building site..." -ForegroundColor Green
hugo --gc

if (-not (Test-Path "public\index.html")) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Deploying to gh-pages..." -ForegroundColor Green
Set-Location public

git init
git add .
git commit -m "deploy: $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
git push https://github.com/ClarkFlyBee/ClarkFlyBee.github.io.git HEAD:gh-pages --force

Set-Location ..
Remove-Item -Recurse -Force public/.git -ErrorAction SilentlyContinue

Write-Host "Done! Visit https://clarkflybee.github.io" -ForegroundColor Cyan
