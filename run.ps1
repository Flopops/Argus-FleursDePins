# TODO: installation of python directly here

$ScriptDir = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
$EnvCreated = $false

$venvPath = Join-Path -Path $ScriptDir -ChildPath "argus_venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    . $venvPath
    $EnvCreated = $true
    Write-Host "`nEnvironment already installed found`n"
}
else {
    Write-Host "`nEnvironment not installed yet`n"
}

function CheckPythonInstallation {
    $python = ""
    
    if (Get-Command "python" -ErrorAction SilentlyContinue) {
        $python = "python"
    }
    elseif (Get-Command "py" -ErrorAction SilentlyContinue) {
        $python = "py -3.10"
    }

    if ($python -eq "") {
        Write-Host -ForegroundColor Red "ERROR: Python3 is not found. Please install Python 3.10 and add it to your PATH."
        Write-Host "You can download Python 3.10 from https://www.python.org/downloads/ : RTFM !!!"
        exit 2
    }

    return $python
}

$python = CheckPythonInstallation

if (-not $EnvCreated) {
    $venvDir = Join-Path -Path $ScriptDir -ChildPath "argus_venv\"
    Invoke-Expression "$python -m venv $venvDir"
    $activatePath = Join-Path -Path $venvDir -ChildPath "Scripts\Activate.ps1"
    if (Test-Path $activatePath) {
        . $activatePath
        Write-Host "Creating a venv environment in $venvDir"
        $requirementsPath = Join-Path -Path $ScriptDir -ChildPath "requirements_win.txt"
        Invoke-Expression "$python -m pip install -r $requirementsPath"
    }
    else {
        Write-Host -ForegroundColor Red "ERROR: Failed to create virtual environment. Aborting..."
        exit 2
    }
}

Write-Host "`nLaunching app...`n"
Invoke-Expression "$python `"$($ScriptDir)\src\app.py`""
deactivate