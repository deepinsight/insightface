param(
    [string]$BuildDirectory,
    [ValidateSet('Release', 'Debug')][string]$Configuration = 'Release',
    [string]$InspireCVSource,
    [switch]$Static,
    [switch]$Samples,
    [switch]$SkipTests,
    [ValidateRange(1, 64)][int]$Jobs = 4,
    [string[]]$CMakeOptions = @()
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT -or ![Environment]::Is64BitProcess) {
    throw 'This build entry point requires Windows and 64-bit PowerShell.'
}
if (!(Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw 'Run from an x64 Visual Studio Native Tools environment.'
}
if ($env:VSCMD_ARG_TGT_ARCH -and $env:VSCMD_ARG_TGT_ARCH -ne 'x64') {
    throw 'This build entry point requires the x64 Native Tools environment.'
}
# Discover the optional CMake/Ninja components from this Visual Studio
# installation instead of assuming a particular machine's installation path.
if ($env:VSINSTALLDIR) {
    $bundledTools = Join-Path $env:VSINSTALLDIR 'Common7\IDE\CommonExtensions\Microsoft\CMake'
    foreach ($relative in @('CMake\bin', 'Ninja')) {
        $directory = Join-Path $bundledTools $relative
        if (Test-Path $directory) { $env:PATH = "$directory;$env:PATH" }
    }
}
$requiredTools = @('cmake.exe', 'ninja.exe')
if (!$SkipTests) { $requiredTools += 'ctest.exe' }
foreach ($tool in $requiredTools) {
    if (!(Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool is required on PATH." }
}
$cmakeVersionOutput = & cmake.exe --version
if ($LASTEXITCODE -ne 0) { throw "Could not read the CMake version: $LASTEXITCODE" }
$versionMatch = [regex]::Match(($cmakeVersionOutput -join "`n"), 'cmake version (\d+\.\d+\.\d+)')
if (!$versionMatch.Success -or [version]$versionMatch.Groups[1].Value -lt [version]'3.20.0') {
    throw 'CMake 3.20 or newer is required.'
}

function Resolve-FileSystemPath([string]$Path, [string]$ParameterName) {
    $provider = $null
    $drive = $null
    $resolved = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath(
        $Path, [ref]$provider, [ref]$drive)
    if ($provider.Name -ne 'FileSystem') { throw "$ParameterName must be a filesystem path." }
    return $resolved
}

$kind = if ($Static) { 'static' } else { 'shared' }
if (!$BuildDirectory) { $BuildDirectory = Join-Path $repo "build\windows-x64-$Configuration-$kind" }
$BuildDirectory = Resolve-FileSystemPath $BuildDirectory 'BuildDirectory'
if (!$SkipTests) {
    foreach ($resource in @('test_res\pack\Pikachu', 'test_res\data\bulk\kun.jpg')) {
        if (!(Test-Path (Join-Path $repo $resource))) {
            throw "Missing $resource. Prepare the existing test resources as described in doc/Windows.md."
        }
    }
    New-Item -ItemType Directory -Force -Path (Join-Path $repo 'test_res\save\video_frames') | Out-Null
}
$options = @('-S', $repo, '-B', $BuildDirectory, '-G', 'Ninja',
    "-DCMAKE_BUILD_TYPE=$Configuration", "-DISF_BUILD_SHARED_LIBS=$(!$Static)",
    "-DISF_BUILD_WITH_SAMPLE=$([bool]$Samples)", "-DISF_BUILD_WITH_TEST=$(!$SkipTests)")
if ($InspireCVSource) {
    $InspireCVSource = Resolve-FileSystemPath $InspireCVSource 'InspireCVSource'
    $options += "-DISF_INSPIRECV_SOURCE_DIR=$InspireCVSource"
}
& cmake.exe @options @CMakeOptions
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed: $LASTEXITCODE" }
& cmake.exe --build $BuildDirectory --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw "Build failed: $LASTEXITCODE" }
if (!$SkipTests) {
    & ctest.exe --test-dir $BuildDirectory --output-on-failure --no-tests=error
    if ($LASTEXITCODE -ne 0) { throw "Tests failed: $LASTEXITCODE" }
}
& cmake.exe --install $BuildDirectory
if ($LASTEXITCODE -ne 0) { throw "SDK installation failed: $LASTEXITCODE" }
$prefixLine = Get-Content (Join-Path $BuildDirectory 'CMakeCache.txt') | Where-Object { $_ -match '^CMAKE_INSTALL_PREFIX:PATH=' } | Select-Object -First 1
if (!$prefixLine) { throw 'Native build did not provide CMAKE_INSTALL_PREFIX.' }
$installedPrefix = $prefixLine -replace '^CMAKE_INSTALL_PREFIX:PATH=', ''
Write-Output "SDK installed to $installedPrefix/InspireFace"
