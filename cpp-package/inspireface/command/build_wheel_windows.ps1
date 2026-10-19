param(
    [string]$PythonExecutable = $env:ISF_WHEEL_PYTHON,
    [string]$SdkDirectory,
    [string]$BuildDirectory,
    [string]$OutputDirectory,
    [string]$InspireCVSource,
    [ValidateRange(1, 64)][int]$Jobs = 4,
    [switch]$TestNative,
    [switch]$NoBuildIsolation,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
function Resolve-FileSystemPath([string]$Path, [string]$ParameterName) {
    $provider = $null
    $drive = $null
    $resolved = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath(
        $Path, [ref]$provider, [ref]$drive)
    if ($provider.Name -ne 'FileSystem') { throw "$ParameterName must be a filesystem path." }
    return $resolved
}

if (!$PythonExecutable) { $PythonExecutable = 'python' }
& $PythonExecutable -c "import platform, struct, sys; assert sys.platform == 'win32' and struct.calcsize('P') == 8 and platform.machine().lower() in ('amd64', 'x86_64'), 'Use Windows x64 Python'; assert sys.version_info >= (3, 7), 'Python 3.7 or newer is required'"
if ($LASTEXITCODE -ne 0) { throw 'A working Windows x64 Python interpreter is required.' }
if (!$OutputDirectory) { $OutputDirectory = Join-Path $repo 'python\dist' }
$OutputDirectory = Resolve-FileSystemPath $OutputDirectory 'OutputDirectory'

if ($SdkDirectory) {
    if ($TestNative -or $InspireCVSource -or $BuildDirectory) {
        throw 'SdkDirectory reuses an installed SDK; omit native build options.'
    }
    $SdkDirectory = Resolve-FileSystemPath $SdkDirectory 'SdkDirectory'
} else {
    if (!$BuildDirectory) { $BuildDirectory = Join-Path $repo 'build\windows-wheel-x64-release' }
    $BuildDirectory = Resolve-FileSystemPath $BuildDirectory 'BuildDirectory'
    $nativeOptions = @{
        BuildDirectory = $BuildDirectory
        Configuration = 'Release'
        Jobs = $Jobs
        SkipTests = !$TestNative
    }
    if ($InspireCVSource) { $nativeOptions.InspireCVSource = $InspireCVSource }
    # Use the existing CPU/MNN shared SDK build and install layout. Wheel
    # packaging omits native tests unless explicitly requested, like Unix.
    & (Join-Path $PSScriptRoot 'build_windows.ps1') @nativeOptions
    $cache = Get-Content (Join-Path $BuildDirectory 'CMakeCache.txt')
    $prefixLine = $cache | Where-Object { $_ -match '^CMAKE_INSTALL_PREFIX:PATH=' } | Select-Object -First 1
    if (!$prefixLine) { throw 'Native build did not provide CMAKE_INSTALL_PREFIX.' }
    $SdkDirectory = $prefixLine -replace '^CMAKE_INSTALL_PREFIX:PATH=', ''
}

$options = @('--repo', $repo, '--sdk', $SdkDirectory, '--output', $OutputDirectory)
if ($NoBuildIsolation) { $options += '--no-build-isolation' }
if ($Force) { $options += '--overwrite' }
& $PythonExecutable (Join-Path $repo 'ci\windows\package_wheel.py') @options
if ($LASTEXITCODE -ne 0) { throw "Windows wheel packaging failed: $LASTEXITCODE" }
