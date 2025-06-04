from pathlib import Path

import mutagen

path = Path(r"W:\media\music\[02] organized\[02] lq\Ambient\Melancholie²\03 - ColdWorld - Winterreise.flac")

file = mutagen.File(path)
file['bpm'] = str(70.0)

file.save()
