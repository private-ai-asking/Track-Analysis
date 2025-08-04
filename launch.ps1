& "V:/Track Analysis/Scripts/activate.ps1"

# Get the current working directory
$originalDir = Get-Location
$newDir = Join-Path -Path $originalDir -ChildPath "track_analysis" -AdditionalChildPath ("components", "md.launcher")

# Add the project's root directory to PYTHONPATH
$env:PYTHONPATH = "$newDir" + ";" + $env:PYTHONPATH

py "./track_analysis/components/md.launcher/md_launcher/components/launcher/launch.py" "./launch_config.json"
