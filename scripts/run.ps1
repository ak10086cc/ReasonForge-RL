param(
    [ValidateSet('info', 'help', 'test', 'pipeline')]
    [string]$Task = 'info',
    [string]$PythonPath = '',
    [string]$Config = 'configs/smoke.yaml',
    [ValidateSet('all', 'env', 'storage', 'prepare', 'baseline', 'sft', 'grpo', 'evaluate', 'ablation', 'analyze')]
    [string]$Phase = 'all',
    [string]$RunDir = '',
    [string]$ResumeCheckpoint = '',
    [double]$EvalMaxSeconds = 0,
    [int]$EvalMaxSamples = 0,
    [string[]]$ArtifactRun = @()
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $PythonPath) {
    $localPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
    if (Test-Path -LiteralPath $localPython -PathType Leaf) { $PythonPath = $localPython }
    else { throw 'No project Python found. Prepare Python 3.12 dependencies or pass -PythonPath.' }
}
$taskPython = (Resolve-Path -LiteralPath $PythonPath).Path
$taskTmp = Join-Path $projectRoot 'tmp'
$envNames = @('TEMP', 'TMP', 'PYTHONDONTWRITEBYTECODE', 'PYTHONUTF8', 'MPLCONFIGDIR', 'HF_HUB_CACHE', 'HF_DATASETS_CACHE', 'HF_XET_CACHE', 'HF_ASSETS_CACHE')
$previousEnvironment = @{}
foreach ($name in $envNames) { $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
Push-Location -LiteralPath $projectRoot
try {
    New-Item -ItemType Directory -Force -Path $taskTmp | Out-Null
    $env:TEMP = $taskTmp
    $env:TMP = $taskTmp
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONUTF8 = '1'
    $env:MPLCONFIGDIR = Join-Path $taskTmp 'matplotlib'
    $projectHfCache = Join-Path $projectRoot '.cache/huggingface'
    $env:HF_HUB_CACHE = Join-Path $projectHfCache 'hub'
    $env:HF_DATASETS_CACHE = Join-Path $projectHfCache 'datasets'
    $env:HF_XET_CACHE = Join-Path $projectHfCache 'xet'
    $env:HF_ASSETS_CACHE = Join-Path $projectHfCache 'assets'
    # Keep authentication settings and the shared pip cache unchanged.
    if ($Task -eq 'info') {
        $inspection = @'
import importlib.metadata as metadata
import json, os, pathlib, shutil, subprocess, sys, sysconfig, tempfile
from huggingface_hub import constants as hub_constants
from datasets import config as dataset_config
print(json.dumps({
    "project": str(pathlib.Path.cwd()),
    "python": sys.executable, "python_version": sys.version.split()[0],
    "dependencies": {name: metadata.version(name) for name in
                     ("torch", "transformers", "trl", "peft", "pytest")},
    "dependency_location": sysconfig.get_path("purelib"),
    "temp": tempfile.gettempdir(), "matplotlib_cache": os.environ["MPLCONFIGDIR"],
    "model_cache": hub_constants.HF_HUB_CACHE,
    "data_cache": str(dataset_config.HF_DATASETS_CACHE),
    "xet_cache": hub_constants.HF_XET_CACHE, "assets_cache": hub_constants.HF_ASSETS_CACHE,
    "pip_cache": subprocess.check_output([sys.executable, "-m", "pip", "cache", "dir"], text=True).strip(),
    "outputs": str(pathlib.Path("outputs/runs").resolve()),
    "disk_free_gib": round(shutil.disk_usage(".").free / 2**30, 2),
}, ensure_ascii=True, indent=2))
'@
        $inspection | & $taskPython -B -
    } elseif ($Task -eq 'test') {
        $testArgs = @('-B', '-m', 'pytest', 'tests', '-q', '--basetemp', 'tmp/pytest-work', '-o', 'cache_dir=tmp/pytest-cache')
        foreach ($run in $ArtifactRun) { $testArgs += @('--artifact-run', $run) }
        & $taskPython @testArgs
    } else {
        $pipelineArgs = @('-B', 'scripts/run_pipeline.py')
        if ($Task -eq 'help') { $pipelineArgs += '--help' }
        else {
            $pipelineArgs += @('--config', $Config, '--phase', $Phase)
            if ($RunDir) { $pipelineArgs += @('--run-dir', $RunDir) }
            if ($ResumeCheckpoint) { $pipelineArgs += @('--resume-checkpoint', $ResumeCheckpoint) }
            if ($PSBoundParameters.ContainsKey('EvalMaxSeconds')) { $pipelineArgs += @('--eval-max-seconds', "$EvalMaxSeconds") }
            if ($PSBoundParameters.ContainsKey('EvalMaxSamples')) { $pipelineArgs += @('--eval-max-samples', "$EvalMaxSamples") }
        }
        & $taskPython @pipelineArgs
    }
    if ($LASTEXITCODE -ne 0) { throw "Task '$Task' failed with exit code $LASTEXITCODE." }
} finally {
    foreach ($name in $envNames) { [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], 'Process') }
    Pop-Location
}
