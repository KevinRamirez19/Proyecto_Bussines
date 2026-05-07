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
    df['IsFree'] = df['Type'] == False
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
 
    cols_comunes  = [c for c in df_raw.columns if c in df_clean.columns]
    nulos_antes   = df_raw[cols_comunes].isnull().sum().sort_values(ascending=False)
    nulos_despues = df_clean[cols_comunes].isnull().sum().reindex(nulos_antes.index).fillna(0)
 
    fig, ax = plt.subplots(figsize=(9, 4))
    style_ax(ax, fig)
    x = range(len(nulos_antes))
    w = 0.38
    bars_a = ax.bar([i - w/2 for i in x], nulos_antes.values,   width=w, color=RED+'bb',    label='Antes',   linewidth=0)
    bars_d = ax.bar([i + w/2 for i in x], nulos_despues.values,  width=w, color=ACCENT+'bb', label='Después', linewidth=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(nulos_antes.index, rotation=35, ha='right', fontsize=8.5, color=MUTED)
    ax.set_ylabel('Valores nulos', color=MUTED, fontsize=9)
 
    # Anotación explicando que los nulos de Rating son intencionales
    rating_idx = list(nulos_antes.index).index('Rating') if 'Rating' in nulos_antes.index else None
    if rating_idx is not None:
        ax.annotate(
            '★ Nulos conservados\nintencionalm. (sin rating)',
            xy=(rating_idx, nulos_despues['Rating']),
            xytext=(rating_idx + 1.2, nulos_despues['Rating'] * 0.85),
            fontsize=7.5, color=ACCENT, style='italic',
            arrowprops=dict(arrowstyle='->', color=ACCENT, lw=0.8)
        )
 
    ax.legend(facecolor=BG3, edgecolor='#ffffff15', labelcolor=TEXT, fontsize=9)
    fig.tight_layout()
    return jsonify({'img': fig_to_b64(fig)})


@app.route('/api/limpieza/graficas/rating')
def grafica_rating():
    df_raw   = pd.read_csv(RAW_PATH)
    df_clean = pd.read_excel(DATA_PATH)
 
    r_antes   = pd.to_numeric(df_raw['Rating'],   errors='coerce').dropna()
    r_despues = pd.to_numeric(df_clean['Rating'], errors='coerce').dropna()
 
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
 
    configs = [
        (r_antes,   RED,    'Antes (CSV crudo)'),
        (r_despues, ACCENT, 'Después (limpio)'),
    ]
 
    for ax, (data, color, label) in zip(axes, configs):
        style_ax(ax, fig)
 
        # Outliers: valores fuera del rango válido (> 5.0)
        outliers = int((data > 5.0).sum())
        # Graficar SOLO datos válidos — el outlier NO entra al histograma
        datos_validos = data[data <= 5.0]
 
        ax.hist(datos_validos, bins=20, color=color+'bb', edgecolor='none', linewidth=0)
        ax.set_xlim(0, 5.5)
        ax.set_title(label, color=TEXT, fontsize=10, pad=8)
        ax.set_xlabel('Rating', color=MUTED, fontsize=9)
        ax.set_ylabel('Apps', color=MUTED, fontsize=9)
        ax.axvline(datos_validos.mean(), color='#ffffff88', linestyle='--', linewidth=1)
 
        # Anotación de outliers solo si los hay (panel "Antes")
        if outliers:
            ax.annotate(
                f'{outliers} outlier(s) > 5.0\n(excluido del histograma)',
                xy=(0.97, 0.97), xycoords='axes fraction',
                ha='right', va='top', fontsize=7.5,
                color=RED, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f871711a',
                        edgecolor=RED+'55', linewidth=0.8)
            )
 
    fig.tight_layout()
    return jsonify({'img': fig_to_b64(fig)})



@app.route('/api/limpieza/graficas/precios')
def grafica_precios():
    df_raw   = pd.read_csv(RAW_PATH)
    df_clean = pd.read_excel(DATA_PATH)
 
    # Solo apps de PAGO en el CSV crudo (excluye precio "0" = gratis)
    raw_prices_pago = df_raw['Price'].dropna().astype(str)
    raw_prices_pago = raw_prices_pago[raw_prices_pago.str.strip() != '0']
 
    def clasificar(v):
        v = v.strip()
        if v.startswith('$'):
            try:
                float(v[1:])
                return 'Con signo $ válido'
            except ValueError:
                return 'Con $ pero inválido'
        try:
            float(v)
            return 'Número sin $'
        except ValueError:
            return 'Texto / NaN'
 
    categorias = raw_prices_pago.apply(clasificar).value_counts()
    colores_cat = [AMBER, '#f59e0b', RED, '#fcd34d'][:len(categorias)]
 
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
 
    # Panel izquierdo: formatos de apps de PAGO en el CSV crudo
    ax0 = axes[0]
    style_ax(ax0, fig)
    bars = ax0.bar(range(len(categorias)), categorias.values,
                   color=colores_cat, edgecolor='none', linewidth=0)
    ax0.set_xticks(range(len(categorias)))
    ax0.set_xticklabels(categorias.index, rotation=20, ha='right', fontsize=8.5, color=MUTED)
    ax0.set_title('Precios antes — apps de pago (formatos CSV)', color=TEXT, fontsize=10, pad=8)
    ax0.set_ylabel('Cantidad de apps', color=MUTED, fontsize=9)
    for bar, val in zip(bars, categorias.values):
        ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha='center', fontsize=8.5, color=MUTED)
 
    # Panel derecho: histograma numérico limpio (solo de pago)
    ax1 = axes[1]
    style_ax(ax1, fig)
    p_despues = pd.to_numeric(df_clean['Price'], errors='coerce').dropna()
    p_despues = p_despues[p_despues > 0]
    ax1.hist(p_despues.clip(upper=30), bins=25, color=ACCENT+'bb', edgecolor='none')
    ax1.set_title('Precios después — apps de pago (float limpio)', color=TEXT, fontsize=10, pad=8)
    ax1.set_xlabel('Precio USD (cap. $30)', color=MUTED, fontsize=9)
    ax1.set_ylabel('Apps', color=MUTED, fontsize=9)
 
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