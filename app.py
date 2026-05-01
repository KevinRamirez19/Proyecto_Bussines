from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

DATA_PATH = os.path.join('data', 'processed', 'googleplaystore_limpieza.xlsx')

def load_data():
    df = pd.read_excel(DATA_PATH)
    df['IsFree'] = ~df['Type']
    df['Last Updated'] = pd.to_datetime(df['Last Updated'])
    df['Year'] = df['Last Updated'].dt.year
    return df

# ── Rutas principales ────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analisis')
def analisis():
    return render_template('analisis.html')

@app.route('/comparacion')
def comparacion():
    return render_template('comparacion.html')

@app.route('/dashboards')
def dashboards():
    return render_template('dashboards.html')

@app.route('/pgc')
def pgc():
    return render_template('pgc.html')

# ── API de datos para las gráficas ───────────────────────────────
@app.route('/api/kpis')
def api_kpis():
    df = load_data()
    return jsonify({
        'total_apps':       int(len(df)),
        'avg_rating':       round(float(df['Rating'].mean()), 2),
        'total_installs':   int(df['Installs'].sum()),
        'free_pct':         round(float(df['IsFree'].mean() * 100), 1),
        'total_categories': int(df['Category'].nunique()),
        'paid_apps':        int((~df['IsFree']).sum()),
    })

@app.route('/api/categorias')
def api_categorias():
    df = load_data()
    top = df['Category'].value_counts().head(15).reset_index()
    top.columns = ['category', 'count']
    top['category'] = top['category'].str.replace('_', ' ').str.title()
    return jsonify(top.to_dict('records'))

@app.route('/api/instalaciones_categoria')
def api_instalaciones_categoria():
    df = load_data()
    top = df.groupby('Category')['Installs'].sum().sort_values(ascending=False).head(10).reset_index()
    top.columns = ['category', 'installs']
    top['category'] = top['category'].str.replace('_', ' ').str.title()
    top['installs_b'] = (top['installs'] / 1e9).round(2)
    return jsonify(top.to_dict('records'))

@app.route('/api/rating_dist')
def api_rating_dist():
    df = load_data()
    bins   = [0, 2, 3, 4, 4.5, 5]
    labels = ['Menos de 2', '2 – 3', '3 – 4', '4 – 4.5', '4.5 – 5']
    dist   = pd.cut(df['Rating'], bins=bins, labels=labels).value_counts().sort_index()
    return jsonify([{'rango': k, 'cantidad': int(v)} for k, v in dist.items()])

@app.route('/api/free_vs_paid')
def api_free_vs_paid():
    df = load_data()
    counts = df['IsFree'].value_counts()
    return jsonify({
        'gratis': int(counts.get(True, 0)),
        'pago':   int(counts.get(False, 0)),
    })

@app.route('/api/por_anio')
def api_por_anio():
    df = load_data()
    yearly = df[df['Year'] >= 2013]['Year'].value_counts().sort_index().reset_index()
    yearly.columns = ['year', 'count']
    return jsonify(yearly.to_dict('records'))

@app.route('/api/top_apps')
def api_top_apps():
    df = load_data()
    top = df.nlargest(10, 'Installs')[['App', 'Installs', 'Rating', 'Category']].copy()
    top['Category'] = top['Category'].str.replace('_', ' ').str.title()
    top['Installs_fmt'] = top['Installs'].apply(lambda x: f"{x/1e9:.1f}B" if x >= 1e9 else f"{x/1e6:.0f}M")
    return jsonify(top.to_dict('records'))

@app.route('/api/content_rating')
def api_content_rating():
    df = load_data()
    cr = df['Content Rating'].value_counts().reset_index()
    cr.columns = ['label', 'count']
    return jsonify(cr.to_dict('records'))

@app.route('/api/tabla')
def api_tabla():
    df = load_data()
    sample = df[['App', 'Category', 'Rating', 'Installs', 'Type', 'Price', 'Content Rating']].copy()
    sample['Category'] = sample['Category'].str.replace('_', ' ').str.title()
    sample['Tipo'] = sample['Type'].map({True: 'Pago', False: 'Gratis'})
    sample = sample.drop(columns=['Type'])
    sample['Installs_fmt'] = sample['Installs'].apply(
        lambda x: f"{x/1e9:.1f}B" if x >= 1e9 else (f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
    sample = sample.head(200)
    return jsonify(sample[['App','Category','Rating','Installs_fmt','Tipo','Price','Content Rating']].to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True)