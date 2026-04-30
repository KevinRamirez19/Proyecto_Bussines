from flask import Flask, render_template


app = Flask(__name__)

@app.route("/")
def inicio ():
    return render_template("index.html")

@app.route("/analisis")
def analisis ():
    return render_template("analisis.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/PGC")
def pgc():
    return render_template("pgc.html")
