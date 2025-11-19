#!/usr/bin/env python3
"""
ZOF_CLI.py
Zero-Of-Function solver CLI implementing:
- Bisection
- Regula-Falsi (False Position)
- Secant
- Newton-Raphson
- Fixed-point iteration
- Modified Secant
Requires: sympy, numpy
"""

import math
import sys
from typing import Callable, Tuple, List, Optional

try:
    import sympy as sp
    import numpy as np
except Exception as e:
    print("Missing dependency. Install with: pip install sympy numpy")
    raise

# ---------- Utility: parse expression to python function ----------
def make_function(expr_str: str, var_symbol='x') -> Tuple[Callable[[float], float], sp.Expr]:
    """
    Returns (callable_f, sympy_expr)
    """
    x = sp.symbols(var_symbol)
    expr = sp.sympify(expr_str)
    f = sp.lambdify(x, expr, modules=["math", "numpy"])
    return f, expr

def make_derivative(expr: sp.Expr, var_symbol='x') -> Callable[[float], float]:
    x = sp.symbols(var_symbol)
    dexpr = sp.diff(expr, x)
    return sp.lambdify(x, dexpr, modules=["math", "numpy"])

# ---------- Methods Implementation ----------
def bisection(f: Callable[[float], float], a: float, b: float, tol: float, max_iter: int):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for bisection.")
    iterations = []
    for n in range(1, max_iter + 1):
        c = 0.5*(a + b)
        fc = f(c)
        est_error = abs(b - a)/2
        iterations.append((n, a, b, c, fa, fb, fc, est_error))
        if abs(fc) == 0 or est_error < tol:
            return c, iterations
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return 0.5*(a+b), iterations

def regula_falsi(f: Callable[[float], float], a: float, b: float, tol: float, max_iter: int):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for Regula-Falsi.")
    iterations = []
    x_prev = None
    for n in range(1, max_iter + 1):
        x = (a*fb - b*fa) / (fb - fa)  # intersection of secant line
        fx = f(x)
        est_error = abs(x - x_prev) if x_prev is not None else None
        iterations.append((n, a, b, x, fa, fb, fx, est_error))
        if abs(fx) == 0 or (est_error is not None and est_error < tol):
            return x, iterations
        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx
        x_prev = x
    return x, iterations

def secant(f: Callable[[float], float], x0: float, x1: float, tol: float, max_iter: int):
    iterations = []
    for n in range(1, max_iter + 1):
        f0, f1 = f(x0), f(x1)
        if (f1 - f0) == 0:
            raise ValueError("Zero denominator in secant formula.")
        x2 = x1 - f1*(x1 - x0)/(f1 - f0)
        est_error = abs(x2 - x1)
        iterations.append((n, x0, x1, x2, f0, f1, est_error))
        if est_error < tol:
            return x2, iterations
        x0, x1 = x1, x2
    return x1, iterations

def newton_raphson(f: Callable[[float], float], df: Callable[[float], float], x0: float, tol: float, max_iter: int):
    iterations = []
    x = x0
    for n in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ValueError("Zero derivative. Newton-Raphson fails.")
        x_next = x - fx/dfx
        est_error = abs(x_next - x)
        iterations.append((n, x, fx, dfx, x_next, est_error))
        if est_error < tol:
            return x_next, iterations
        x = x_next
    return x, iterations

def fixed_point(g: Callable[[float], float], x0: float, tol: float, max_iter: int):
    iterations = []
    x = x0
    for n in range(1, max_iter + 1):
        x_next = g(x)
        est_error = abs(x_next - x)
        iterations.append((n, x, x_next, est_error))
        if est_error < tol:
            return x_next, iterations
        x = x_next
    return x, iterations

def modified_secant(f: Callable[[float], float], x0: float, delta: float, tol: float, max_iter: int):
    iterations = []
    x = x0
    for n in range(1, max_iter + 1):
        fx = f(x)
        denom = f(x + delta*x) - fx
        if denom == 0:
            raise ValueError("Zero denominator in modified secant.")
        x_next = x - fx * delta * x / denom
        est_error = abs(x_next - x)
        iterations.append((n, x, fx, delta, x_next, est_error))
        if est_error < tol:
            return x_next, iterations
        x = x_next
    return x, iterations

# ---------- Printing helpers ----------
def print_bisection_iters(iters):
    print("\nIteration | a\t\t b\t\t c\t\t f(c)\t\t error")
    for row in iters:
        n,a,b,c,fa,fb,fc,err = row
        print(f"{n:9d} | {a:.6g}\t {b:.6g}\t {c:.6g}\t {fc:.6g}\t {err:.6g}")

def print_regula_iters(iters):
    print("\nIter | a\t b\t root\t f(root)\t error")
    for row in iters:
        n,a,b,x,fa,fb,fx,err = row
        err_str = f"{err:.6g}" if err is not None else "N/A"
        print(f"{n:4d} | {a:.6g}\t {b:.6g}\t {x:.6g}\t {fx:.6g}\t {err_str}")

def print_secant_iters(iters):
    print("\nIter | x0\t x1\t x2\t f(x1)\t error")
    for row in iters:
        n,x0,x1,x2,f0,f1,err = row
        print(f"{n:4d} | {x0:.6g}\t {x1:.6g}\t {x2:.6g}\t {f1:.6g}\t {err:.6g}")

def print_newton_iters(iters):
    print("\nIter | x\t f(x)\t f'(x)\t x_next\t error")
    for row in iters:
        n,x,fx,dfx,x_next,err = row
        print(f"{n:4d} | {x:.6g}\t {fx:.6g}\t {dfx:.6g}\t {x_next:.6g}\t {err:.6g}")

def print_fixed_iters(iters):
    print("\nIter | x\t x_next\t error")
    for row in iters:
        n,x,x_next,err = row
        print(f"{n:4d} | {x:.6g}\t {x_next:.6g}\t {err:.6g}")

def print_modified_secant_iters(iters):
    print("\nIter | x\t f(x)\t delta\t x_next\t error")
    for row in iters:
        n,x,fx,delta,x_next,err = row
        print(f"{n:4d} | {x:.6g}\t {fx:.6g}\t {delta:.6g}\t {x_next:.6g}\t {err:.6g}")

# ---------- CLI interaction ----------
def main():
    print("ZOF_CLI — Zero of Functions Solver")
    print("Supported methods: bisection, regula, secant, newton, fixed, modified_secant")
    method = input("Choose method: ").strip().lower()
    expr_str = input("Enter f(x) (e.g. x**3 - x - 2): ").strip()
    f, expr = make_function(expr_str)
    df = None
    if method == "newton":
        df = make_derivative(expr)

    tol = float(input("Tolerance (e.g. 1e-6): ") or 1e-6)
    max_iter = int(input("Max iterations (e.g. 50): ") or 50)

    try:
        if method == "bisection":
            a = float(input("Left endpoint a: "))
            b = float(input("Right endpoint b: "))
            root, iters = bisection(f, a, b, tol, max_iter)
            print_bisection_iters(iters)
            print(f"\nEstimated root: {root:.12g}, last error estimate: {iters[-1][-1]:.6g}, iterations: {len(iters)}")

        elif method == "regula":
            a = float(input("Left endpoint a: "))
            b = float(input("Right endpoint b: "))
            root, iters = regula_falsi(f, a, b, tol, max_iter)
            print_regula_iters(iters)
            print(f"\nEstimated root: {root:.12g}, iterations: {len(iters)}")

        elif method == "secant":
            x0 = float(input("Initial guess x0: "))
            x1 = float(input("Initial guess x1: "))
            root, iters = secant(f, x0, x1, tol, max_iter)
            print_secant_iters(iters)
            print(f"\nEstimated root: {root:.12g}, iterations: {len(iters)}")

        elif method == "newton":
            x0 = float(input("Initial guess x0: "))
            root, iters = newton_raphson(f, df, x0, tol, max_iter)
            print_newton_iters(iters)
            print(f"\nEstimated root: {root:.12g}, iterations: {len(iters)}")

        elif method == "fixed":
            g_str = input("Enter g(x) for fixed-point (x = g(x)): ")
            g, gexpr = make_function(g_str)
            x0 = float(input("Initial guess x0: "))
            root, iters = fixed_point(g, x0, tol, max_iter)
            print_fixed_iters(iters)
            print(f"\nEstimated fixed point: {root:.12g}, iterations: {len(iters)}")

        elif method == "modified_secant":
            x0 = float(input("Initial guess x0: "))
            delta = float(input("Delta (e.g. 1e-3): ") or 1e-3)
            root, iters = modified_secant(f, x0, delta, tol, max_iter)
            print_modified_secant_iters(iters)
            print(f"\nEstimated root: {root:.12g}, iterations: {len(iters)}")

        else:
            print("Unknown method.")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
