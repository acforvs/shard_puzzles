"""uv run check.py 0              check section 00 (0a, 0b and 0c)
uv run check.py                check everything
uv run check.py 0a --solution  check the reference solution
uv run check.py 0a --hint      the first hint (--hint 2, ...); --scaffold for a skeleton with checkpoints"""

import sys

from shardpuzzles.cli import main

sys.exit(main())
