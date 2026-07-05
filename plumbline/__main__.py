"""Enable `python -m plumbline …` (mirrors the `plumbline` console script)."""

from plumbline.cli import main

raise SystemExit(main())
