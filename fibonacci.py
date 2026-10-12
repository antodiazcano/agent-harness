#!/usr/bin/env python3
import sys

def fibonacci(n:int)->int:
    """Return the nth Fibonacci number, with F(0)=0, F(1)=1."""
    if n<0:
        raise ValueError("n must be non‑negative")
    a,b=0,1
    for _ in range(n):
        a,b=b,a+b
    return a

if __name__=='__main__':
    if len(sys.argv)!=2:
        print("Usage: python fibonacci.py <non-negative integer>")
        sys.exit(1)
    try:
        n=int(sys.argv[1])
    except ValueError:
        print("Error: argument must be an integer")
        sys.exit(1)
    if n<0:
        print("Error: n must be non-negative")
        sys.exit(1)
    print(fibonacci(n))
