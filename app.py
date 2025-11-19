# app.py
from flask import Flask, render_template, request, Markup
import sympy as sp
from ZOF_CLI import (make_function, make_derivative,
                     bisection, regula_falsi, secant,
                     newton_raphson, fixed_point, modified_secant,
                     print_bisection_iters)  # re-uses functions (ensure same folder)

app = Flask(__name__)

def format_table_from_iters(method, iters):
    # Build HTML table from iterations (simple)
    rows = []
    if method == "bisection":
        header = ["n","a","b","c","f(c)","error"]
        for r in iters:
            n,a,b,c,fa,fb,fc,err = r
            rows.append([n, a, b, c, fc, err])
    elif method == "regula":
        header = ["n","a","b","root","f(root)","error"]
        for r in iters:
            n,a,b,x,fa,fb,fx,err = r
            rows.append([n,a,b,x,fx,err])
    elif method == "secant":
        header = ["n","x0","x1","x2","f(x1)","error"]
        for r in iters:
            n,x0,x1,x2,f0,f1,err = r
            rows.append([n,x0,x1,x2,f1,err])
    elif method == "newton":
        header = ["n","x","f(x)","f'(x)","x_next","error"]
        for r in iters:
            n,x,fx,dfx,x_next,err = r
            rows.append([n,x,fx,dfx,x_next,err])
    elif method == "fixed":
        header = ["n","x","x_next","error"]
        for r in iters:
            n,x,x_next,err = r
            rows.append([n,x,x_next,err])
    elif method == "modified_secant":
        header = ["n","x","f(x)","delta","x_next","error"]
        for r in iters:
            n,x,fx,delta,x_next,err = r
            rows.append([n, x, fx, delta, x_next, err])
    else:
        header = ["Iter","values"]
    # create html
    html = "<table border='1' cellpadding='6'><tr>"
    for h in header:
        html += f"<th>{h}</th>"
    html += "</tr>"
    for row in rows:
        html += "<tr>"
        for v in row:
            html += f"<td>{v}</td>"
        html += "</tr>"
    html += "</table>"
    return html

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)

@app.route("/solve", methods=["POST"])
def solve():
    method = request.form.get("method")
    fexpr = request.form.get("fexpr")
    tol = float(request.form.get("tol") or 1e-6)
    max_iter = int(request.form.get("max_iter") or 50)

    try:
        f, expr = make_function(fexpr)
        if method == "bisection":
            a = float(request.form.get("a"))
            b = float(request.form.get("b"))
            root, iters = bisection(f, a, b, tol, max_iter)
        elif method == "regula":
            a = float(request.form.get("a"))
            b = float(request.form.get("b"))
            root, iters = regula_falsi(f, a, b, tol, max_iter)
        elif method == "secant":
            x0 = float(request.form.get("x0"))
            x1 = float(request.form.get("x1"))
            root, iters = secant(f, x0, x1, tol, max_iter)
        elif method == "newton":
            x0 = float(request.form.get("x0"))
            df = make_derivative(expr)
            root, iters = newton_raphson(f, df, x0, tol, max_iter)
        elif method == "fixed":
            gexpr = request.form.get("gexpr")
            g, _ = make_function(gexpr)
            x0 = float(request.form.get("x0"))
            root, iters = fixed_point(g, x0, tol, max_iter)
        elif method == "modified_secant":
            x0 = float(request.form.get("x0"))
            delta = float(request.form.get("delta") or 1e-3)
            root, iters = modified_secant(f, x0, delta, tol, max_iter)
        else:
            return render_template("index.html", result="Unknown method")
        table_html = format_table_from_iters(method, iters)
        summary = f"Estimated root: {root} (iterations: {len(iters)})"
        return render_template("index.html", result=Markup(table_html), summary=summary)
    except Exception as e:
        return render_template("index.html", result=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
