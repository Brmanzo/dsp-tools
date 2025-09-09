#!/usr/bin/env python3
# DSP-tools/main.py
import sys
from tools.tools import get_function

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        return
    get_function(sys.argv)

if __name__ == "__main__":
    main()
