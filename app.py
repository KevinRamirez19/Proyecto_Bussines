from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64
app = Flask(__name__)

DATA_PATH = os.path.join('data', 'processed', 'googleplaystore_limpieza.xlsx')
RAW_PATH  = os.path.join('data', 'raw', 'googleplaystore.csv')

def load_data():
    df = pd.read_excel(DATA_PATH)
    df['IsFree'] = ~df['Type']          # True = es de pago → ojo al mapear
    df['Last Updated'] = pd.to_datetime(df['Last Updated'])
    df['Year'] = df['Last Updated'].dt.year
    return df

# ── Rutas principales ────────────────────────────────────────────
@app.route('/')
def index():      return render_template('index.html')

@app.route('/analisis')
def analisis():   return render_template('analisis.html')

@app.route('/comparacion')
def comparacion(): return render_template('comparacion.html')

@app.route('/dashboards')
def dashboards(): return render_template('dashboards.html')

@app.route('/pgc')
def pgc():        return render_template('pgc.html')

@app.route('/limpieza')                         
def limpieza():   return render_template('limpieza.html')


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
    top['Installs_fmt'] = top['Installs'].apply(
        lambda x: f"{x/1e9:.1f}B" if x >= 1e9 else f"{x/1e6:.0f}M")
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

    sample['Tipo'] = sample['Type'].apply(
        lambda x: 'Gratis' if x is False or x == 'Free' or x == False else 'Pago'
    )
    sample = sample.drop(columns=['Type'])

    sample['Installs_fmt'] = sample['Installs'].apply(
        lambda x: f"{x/1e9:.1f}B" if x >= 1e9 else (
                f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
    )

    
    sample = sample.replace({np.nan: None})

    return jsonify(
        sample[['App','Category','Rating','Installs_fmt','Tipo','Price','Content Rating']]
        .to_dict('records')
    )


@app.route('/api/limpieza/stats')
def api_limpieza_stats():
    """Estadísticas comparativas antes/después de la limpieza"""
    try:
        df_raw   = pd.read_csv(RAW_PATH)
    except Exception:
        df_raw   = None

    df_clean = pd.read_excel(DATA_PATH)

    stats = {
        'antes': {
            'filas':      int(len(df_raw)) if df_raw is not None else '—',
            'columnas':   int(len(df_raw.columns)) if df_raw is not None else '—',
            'nulos':      int(df_raw.isnull().sum().sum()) if df_raw is not None else '—',
            'duplicados': int(df_raw.duplicated().sum()) if df_raw is not None else '—',
        },
        'despues': {
            'filas':      int(len(df_clean)),
            'columnas':   int(len(df_clean.columns)),
            'nulos':      int(df_clean.isnull().sum().sum()),
            'duplicados': int(df_clean.duplicated().sum()),
        }
    }
    return jsonify(stats)

@app.route('/api/limpieza/muestra')
def api_limpieza_muestra():
    """Muestra 8 filas del antes y después para comparar"""
    try:
        df_raw = pd.read_csv(RAW_PATH)
        antes  = df_raw[['App','Category','Rating','Installs','Type','Price']].head(8).replace({np.nan: None}).to_dict('records')
    except Exception:
        antes = []

    df_clean = pd.read_excel(DATA_PATH)
    cols_clean = ['App','Category','Rating','Installs','Type','Price']
    cols_ok    = [c for c in cols_clean if c in df_clean.columns]
    despues    = df_clean[cols_ok].head(8).replace({np.nan: None}).to_dict('records')

    return jsonify({'antes': antes, 'despues': despues})

# ── Paleta light mode ─────────────────────────────────────────────
ACCENT  = '#16a34a'
RED     = '#dc2626'
BLUE    = '#0284c7'
AMBER   = '#d97706'
PURPLE  = '#7c3aed'
MUTED   = '#5a7a65'
BG2     = '#ffffff'
BG3     = '#e8f5ee'
TEXT    = '#0d1f16'

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor='#f0faf4',
                edgecolor='none', dpi=110)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64

def style_ax(ax, fig):
    fig.patch.set_facecolor('#f0faf4')
    ax.set_facecolor('#ffffff')
    ax.tick_params(colors='#5a7a65', labelsize=9)
    ax.xaxis.label.set_color('#5a7a65')
    ax.yaxis.label.set_color('#5a7a65')
    for spine in ax.spines.values():
        spine.set_edgecolor('#ccddcc')
    ax.grid(axis='y', color='#d4edda', linewidth=0.8)
    ax.grid(axis='x', color='#d4edda', linewidth=0.8)

@app.route('/api/limpieza/graficas/nulos')
def grafica_nulos():
    df_raw   = pd.read_csv(RAW_PATH)
    df_clean = pd.read_excel(DATA_PATH)

    nulos_antes  = df_raw.isnull().sum().sort_values(ascending=False)
    nulos_despues = df_clean.reindex(columns=df_raw.columns).isnull().sum().reindex(nulos_antes.index).fillna(0)

    fig, ax = plt.subplots(figsize=(9, 4))
    style_ax(ax, fig)
    x = range(len(nulos_antes))
    w = 0.38
    ax.bar([i - w/2 for i in x], nulos_antes.values,  width=w, color=RED+'bb',   label='Antes',   linewidth=0)
    ax.bar([i + w/2 for i in x], nulos_despues.values, width=w, color=ACCENT+'bb', label='Después', linewidth=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(nulos_antes.index, rotation=35, ha='right', fontsize=8.5, color=MUTED)
    ax.set_ylabel('Valores nulos', color=MUTED, fontsize=9)
    ax.legend(facecolor=BG3, edgecolor='#ffffff15', labelcolor=TEXT, fontsize=9)
    fig.tight_layout()
    return jsonify({'img': fig_to_b64(fig)})


@app.route('/api/limpieza/graficas/rating')
def grafica_rating():
    df_raw   = pd.read_csv(RAW_PATH)
    df_clean = pd.read_excel(DATA_PATH)

    r_antes  = pd.to_numeric(df_raw['Rating'],   errors='coerce').dropna()
    r_despues = pd.to_numeric(df_clean['Rating'], errors='coerce').dropna()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, data, color, label in zip(
        axes,
        [r_antes, r_despues],
        [RED, ACCENT],
        ['Antes (CSV crudo)', 'Después (limpio)']
    ):
        style_ax(ax, fig)
        ax.hist(data, bins=20, color=color+'bb', edgecolor='none', linewidth=0)
        ax.set_title(label, color=TEXT, fontsize=10, pad=8)
        ax.set_xlabel('Rating', color=MUTED, fontsize=9)
        ax.set_ylabel('Apps', color=MUTED, fontsize=9)
        ax.axvline(data.mean(), color='#ffffff55', linestyle='--', linewidth=1)

    fig.tight_layout()
    return jsonify({'img': fig_to_b64(fig)})


@app.route('/api/limpieza/graficas/precios')
def grafica_precios():
    df_raw   = pd.read_csv(RAW_PATH)
    df_clean = pd.read_excel(DATA_PATH)

    # Antes: limpiar manualmente para comparar
    p_antes = df_raw['Price'].str.replace('$', '', regex=False)
    p_antes = pd.to_numeric(p_antes, errors='coerce').dropna()
    p_antes = p_antes[p_antes > 0]  # solo de pago

    p_despues = pd.to_numeric(df_clean['Price'], errors='coerce').dropna()
    p_despues = p_despues[p_despues > 0]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, data, color, label in zip(
        axes,
        [p_antes, p_despues],
        [AMBER, ACCENT],
        ['Precios antes (string)', 'Precios después (float)']
    ):
        style_ax(ax, fig)
        ax.hist(data.clip(upper=30), bins=25, color=color+'bb', edgecolor='none')
        ax.set_title(label, color=TEXT, fontsize=10, pad=8)
        ax.set_xlabel('Precio USD (cap. $30)', color=MUTED, fontsize=9)
        ax.set_ylabel('Apps', color=MUTED, fontsize=9)

    fig.tight_layout()
    return jsonify({'img': fig_to_b64(fig)})


@app.route('/api/limpieza/graficas/categorias')
def grafica_categorias():
    df_raw   = pd.read_csv(RAW_PATH)
    df_clean = pd.read_excel(DATA_PATH)

    top_antes  = df_raw['Category'].value_counts().head(12)
    top_despues = df_clean['Category'].value_counts().reindex(top_antes.index).fillna(0)

    fig, ax = plt.subplots(figsize=(9, 5))
    style_ax(ax, fig)
    x = range(len(top_antes))
    w = 0.38
    ax.bar([i - w/2 for i in x], top_antes.values,   width=w, color=BLUE+'bb',   label='Antes',   linewidth=0)
    ax.bar([i + w/2 for i in x], top_despues.values,  width=w, color=ACCENT+'bb', label='Después', linewidth=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [c.replace('_', ' ').title() for c in top_antes.index],
        rotation=35, ha='right', fontsize=8, color=MUTED
    )
    ax.set_ylabel('Cantidad de apps', color=MUTED, fontsize=9)
    ax.legend(facecolor=BG3, edgecolor='#ffffff15', labelcolor=TEXT, fontsize=9)
    fig.tight_layout()
    return jsonify({'img': fig_to_b64(fig)})

if __name__ == '__main__':
    app.run(debug=True)