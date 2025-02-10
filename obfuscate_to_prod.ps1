Remove-Item -Path ".\prod\" -Recurse -Force -ErrorAction SilentlyContinue


.\argus_venv\Scripts\Activate.ps1
pyarmor.exe gen -O prod .\src\
Move-Item ".\prod\pyarmor_runtime_000000" -Destination ".\prod\src\"
Copy-Item ".\pretrained-models" -Destination ".\prod\" -Recurse
Copy-Item ".\LISEZMOI.md", ".\run.ps1", ".\requirements_win.txt" -Destination ".\prod\"