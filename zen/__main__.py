"""Allow running ZEN via `python -m zen run file.zen`."""

import sys
from zen.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
