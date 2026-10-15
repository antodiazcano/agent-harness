#!/usr/bin/env python3

import sys


def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number.

    Parameters
    ----------
    n: int
        Non‑negative index in the Fibonacci sequence.

    Returns
    -------
    int
        Fibonacci number at position n.
    """
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non‑negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: fibonacci.py <n>", file=sys.stderr)
        sys.exit(1)
    try:
        n = int(sys.argv[1])
    except ValueError:
        print("Error: n must be an integer", file=sys.stderr)
        sys.exit(1)
    if n < 0:
        print("Error: n must be non‑negative", file=sys.stderr)
        sys.exit(1)
    print(fibonacci(n))
