from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, classification_report
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64
warnings.filterwarnings('ignore')
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

@app.route('/predictivo')
def predictivo():
    return render_template('predictivo.html')

# ── Helpers de modelos (se entrenan una sola vez por request) ───────────────
def preparar_features(df):
    """Prepara el dataframe para ML: codifica categorías, limpia nulos."""
    d = df.copy()
    d['Category_enc']       = LabelEncoder().fit_transform(d['Category'].astype(str))
    d['ContentRating_enc']  = LabelEncoder().fit_transform(d['Content Rating'].astype(str))
    d['IsFree_int']         = d['IsFree'].astype(int)
    d['Size_MB']            = pd.to_numeric(d.get('Size_MB', pd.Series(dtype=float)), errors='coerce').fillna(d.get('Size_MB', pd.Series(dtype=float)).median() if 'Size_MB' in d.columns else 10)
    d['Price']              = pd.to_numeric(d['Price'], errors='coerce').fillna(0)
    d['Reviews']            = pd.to_numeric(d.get('Reviews', pd.Series(dtype=float)), errors='coerce').fillna(0)
    return d

FEATURES_RATING   = ['Category_enc', 'ContentRating_enc', 'IsFree_int', 'Price', 'Reviews']
FEATURES_INSTALLS = ['Category_enc', 'ContentRating_enc', 'IsFree_int', 'Price', 'Rating']
FEATURES_CLUSTER  = ['Rating', 'IsFree_int', 'Price']

# ── API: métricas generales de los 3 modelos ────────────────────────────────
@app.route('/api/predictivo/metricas')
def api_predictivo_metricas():
    df = load_data()
    d  = preparar_features(df)

    # ── 1. Regresión: predicción de Rating ──────────────────────────────────
    mask_r = d['Rating'].notna()
    X_r = d.loc[mask_r, FEATURES_RATING].fillna(0)
    y_r = d.loc[mask_r, 'Rating']
    X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
    rf_rating = RandomForestRegressor(n_estimators=80, random_state=42)
    rf_rating.fit(X_tr, y_tr)
    r2   = round(r2_score(y_te, rf_rating.predict(X_te)), 3)
    rmse = round(float(np.sqrt(((y_te - rf_rating.predict(X_te))**2).mean())), 3)

    # ── 2. Clasificación: nivel de instalaciones ────────────────────────────
    def nivel_installs(x):
        if x >= 10_000_000: return 'Alto (10M+)'
        elif x >= 100_000:  return 'Medio (100K–10M)'
        else:               return 'Bajo (<100K)'

    d['nivel'] = d['Installs'].apply(nivel_installs)
    mask_i = d['Rating'].notna()
    X_i = d.loc[mask_i, FEATURES_INSTALLS].fillna(0)
    y_i = d.loc[mask_i, 'nivel']
    X_tri, X_tei, y_tri, y_tei = train_test_split(X_i, y_i, test_size=0.2, random_state=42)
    train_size = int(len(X_tr))
    test_size  = int(len(X_te))
    rf_inst = RandomForestClassifier(n_estimators=80, random_state=42)
    rf_inst.fit(X_tri, y_tri)
    acc = round(accuracy_score(y_tei, rf_inst.predict(X_tei)), 3)

    # ── 3. Clustering K-Means ───────────────────────────────────────────────
    X_cl = d[FEATURES_CLUSTER].fillna(0)
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_cl)
    km      = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels  = km.fit_predict(X_sc)
    d['cluster'] = labels
    cluster_info = []
    for cl in sorted(d['cluster'].unique()):
        g = d[d['cluster'] == cl]
        cluster_info.append({
            'id':        int(cl),
            'tamanio':   int(len(g)),
            'rating':    round(float(g['Rating'].mean()), 2),
            'installs':  int(g['Installs'].median()),
            'free_pct':  round(float(g['IsFree'].mean() * 100), 1),
        })

    # ── Importancias del modelo de Rating ──────────────────────────────────
    imp = dict(zip(FEATURES_RATING, rf_rating.feature_importances_.round(3).tolist()))

    return jsonify({
        'rating_model':   {'r2': r2, 'rmse': rmse, 'importancias': imp},
        'installs_model': {'accuracy': acc},
        'clusters':       cluster_info,
        'train_size': train_size,
        'test_size':  test_size
    })

# ── API: gráfica de correlaciones ──────────────────────────────────────────
@app.route('/api/predictivo/graficas/correlacion')
def grafica_correlacion():
    df = load_data()
    d  = preparar_features(df)
    cols = ['Rating', 'Installs', 'Price', 'Reviews', 'IsFree_int']
    corr = d[cols].corr().round(2)

    fig, ax = plt.subplots(figsize=(6, 5))
    style_ax(ax, fig)
    im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    labels_es = ['Rating', 'Installs', 'Precio', 'Reseñas', 'Es Gratis']
    ax.set_xticklabels(labels_es, rotation=30, ha='right', fontsize=9, color=MUTED)
    ax.set_yticklabels(labels_es, fontsize=9, color=MUTED)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, str(corr.values[i, j]),
                    ha='center', va='center', fontsize=9,
                    color='#0d1f16' if abs(corr.values[i, j]) < 0.6 else 'white')
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Matriz de correlación', color=TEXT, fontsize=11, pad=10)
    fig.tight_layout()
    return jsonify({'img': fig_to_b64(fig)})

# ── API: gráfica importancia de variables ──────────────────────────────────
@app.route('/api/predictivo/graficas/importancia')
def grafica_importancia():
    df = load_data()
    d  = preparar_features(df)

    mask = d['Rating'].notna()
    X = d.loc[mask, FEATURES_RATING].fillna(0)
    y = d.loc[mask, 'Rating']
    rf = RandomForestRegressor(n_estimators=80, random_state=42)
    rf.fit(X, y)

    nombres = ['Categoría', 'Clasificación\ncontenido', 'Es Gratis', 'Precio', 'Reseñas']
    imp     = rf.feature_importances_
    orden   = np.argsort(imp)

    fig, ax = plt.subplots(figsize=(7, 4))
    style_ax(ax, fig)
    bars = ax.barh([nombres[i] for i in orden], imp[orden],
                   color=ACCENT + 'bb', edgecolor='none', linewidth=0)
    for bar, val in zip(bars, imp[orden]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9, color=MUTED)
    ax.set_xlabel('Importancia relativa', color=MUTED, fontsize=9)
    ax.set_title('Variables más influyentes en el Rating', color=TEXT, fontsize=11, pad=10)
    ax.set_xlim(0, imp.max() + 0.08)
    fig.tight_layout()
    return jsonify({'img': fig_to_b64(fig)})

# ── API: gráfica clusters ──────────────────────────────────────────────────
@app.route('/api/predictivo/graficas/clusters')
def grafica_clusters():
    df = load_data()
    d  = preparar_features(df)

    X_cl  = d[FEATURES_CLUSTER].fillna(0)
    sc    = StandardScaler()
    X_sc  = sc.fit_transform(X_cl)
    km    = KMeans(n_clusters=3, random_state=42, n_init=10)
    d['cluster'] = km.fit_predict(X_sc)

    colores = [ACCENT + 'cc', BLUE + 'cc', AMBER + 'cc']
    nombres = ['Grupo A', 'Grupo B', 'Grupo C']

    fig, ax = plt.subplots(figsize=(7, 4))
    style_ax(ax, fig)
    for cl in range(3):
        g = d[d['cluster'] == cl].sample(min(400, len(d[d['cluster'] == cl])), random_state=42)
        ax.scatter(g['Rating'], np.log1p(g['Installs']),
                   color=colores[cl], alpha=0.5, s=18,
                   label=f'{nombres[cl]} (n={len(d[d["cluster"]==cl])})', edgecolors='none')
    ax.set_xlabel('Rating', color=MUTED, fontsize=9)
    ax.set_ylabel('log(Installs + 1)', color=MUTED, fontsize=9)
    ax.set_title('Clusters K-Means: Rating vs Instalaciones', color=TEXT, fontsize=11, pad=10)
    ax.legend(facecolor=BG3, edgecolor='#ccddcc', labelcolor=TEXT, fontsize=9)
    fig.tight_layout()
    return jsonify({'img': fig_to_b64(fig)})

# ── API: predicción en tiempo real desde el formulario ─────────────────────
@app.route('/api/predictivo/predecir', methods=['POST'])
def api_predecir():
    from flask import request
    data = request.json

    df = load_data()
    d  = preparar_features(df)

    # Encoders entrenados con los datos reales
    le_cat  = LabelEncoder().fit(d['Category'].astype(str))
    le_cr   = LabelEncoder().fit(d['Content Rating'].astype(str))

    # Entrada del usuario
    cat_input = data.get('category', 'TOOLS')
    cr_input  = data.get('content_rating', 'Everyone')
    is_free   = int(data.get('is_free', 1))
    price     = float(data.get('price', 0))
    reviews   = float(data.get('reviews', 1000))
    rating    = float(data.get('rating', 4.0))

    # Encode — si el valor no está en el encoder, usar 0
    try:
        cat_enc = int(le_cat.transform([cat_input])[0])
    except:
        cat_enc = 0
    try:
        cr_enc = int(le_cr.transform([cr_input])[0])
    except:
        cr_enc = 0

    # ── Modelo Rating ──────────────────────────────────────────────────────
    mask_r = d['Rating'].notna()
    X_r = d.loc[mask_r, FEATURES_RATING].fillna(0)
    y_r = d.loc[mask_r, 'Rating']
    rf_rating = RandomForestRegressor(n_estimators=80, random_state=42)
    rf_rating.fit(X_r, y_r)
    x_rating = [[cat_enc, cr_enc, is_free, price, reviews]]
    pred_rating = round(float(rf_rating.predict(x_rating)[0]), 2)
    pred_rating = min(5.0, max(1.0, pred_rating))

    # ── Modelo Installs ────────────────────────────────────────────────────
    def nivel_installs(x):
        if x >= 10_000_000: return 'Alto (10M+)'
        elif x >= 100_000:  return 'Medio (100K–10M)'
        else:               return 'Bajo (<100K)'

    d['nivel'] = d['Installs'].apply(nivel_installs)
    mask_i = d['Rating'].notna()
    X_i = d.loc[mask_i, FEATURES_INSTALLS].fillna(0)
    y_i = d.loc[mask_i, 'nivel']
    rf_inst = RandomForestClassifier(n_estimators=80, random_state=42)
    rf_inst.fit(X_i, y_i)
    x_inst = [[cat_enc, cr_enc, is_free, price, rating]]
    pred_nivel  = rf_inst.predict(x_inst)[0]
    pred_probas = rf_inst.predict_proba(x_inst)[0]
    clases      = rf_inst.classes_.tolist()
    probas_dict = {c: round(float(p) * 100, 1) for c, p in zip(clases, pred_probas)}

    # ── Cluster ────────────────────────────────────────────────────────────
    X_cl = d[FEATURES_CLUSTER].fillna(0)
    sc   = StandardScaler()
    X_sc = sc.fit_transform(X_cl)
    km   = KMeans(n_clusters=3, random_state=42, n_init=10)
    km.fit(X_sc)
    x_cl_raw = [[rating, is_free, price]]
    x_cl_sc  = sc.transform(x_cl_raw)
    cluster  = int(km.predict(x_cl_sc)[0])

    return jsonify({
        'rating_pred':   pred_rating,
        'nivel_installs': pred_nivel,
        'probabilidades': probas_dict,
        'cluster':        cluster,
    })

@app.route('/api/predictivo/graficas/prediccion')
def grafica_prediccion():
    df = load_data()
    d  = preparar_features(df)

    mask = d['Rating'].notna()
    X = d.loc[mask, FEATURES_RATING].fillna(0)
    y = d.loc[mask, 'Rating']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=80, random_state=42)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)

    fig, ax = plt.subplots(figsize=(6, 5))
    style_ax(ax, fig)
    ax.scatter(y_te, y_pred, color=ACCENT + '55', s=12, edgecolors='none')
    ax.plot([1, 5], [1, 5], color=RED, linewidth=1.2, linestyle='--', label='Predicción perfecta')
    ax.set_xlabel('Rating real', color=MUTED, fontsize=9)
    ax.set_ylabel('Rating predicho', color=MUTED, fontsize=9)
    ax.set_title('Rating real vs predicho (conjunto test)', color=TEXT, fontsize=11, pad=10)
    ax.legend(facecolor=BG3, edgecolor='#ccddcc', labelcolor=TEXT, fontsize=9)
    ax.set_xlim(1, 5.2)
    ax.set_ylim(1, 5.2)
    fig.tight_layout()
    return jsonify({'img': fig_to_b64(fig)})


@app.route('/api/predictivo/graficas/residuos')
def grafica_residuos():
    df = load_data()
    d  = preparar_features(df)

    mask = d['Rating'].notna()
    X = d.loc[mask, FEATURES_RATING].fillna(0)
    y = d.loc[mask, 'Rating']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=80, random_state=42)
    rf.fit(X_tr, y_tr)
    residuos = y_te - rf.predict(X_te)

    fig, ax = plt.subplots(figsize=(6, 5))
    style_ax(ax, fig)
    ax.hist(residuos, bins=30, color=BLUE + 'bb', edgecolor='none', linewidth=0)
    ax.axvline(0, color=RED, linewidth=1.2, linestyle='--', label='Error = 0')
    ax.axvline(residuos.mean(), color=AMBER, linewidth=1,
               linestyle=':', label=f'Media = {residuos.mean():.3f}')
    ax.set_xlabel('Error (real − predicho)', color=MUTED, fontsize=9)
    ax.set_ylabel('Frecuencia', color=MUTED, fontsize=9)
    ax.set_title('Distribución de residuos del modelo', color=TEXT, fontsize=11, pad=10)
    ax.legend(facecolor=BG3, edgecolor='#ccddcc', labelcolor=TEXT, fontsize=9)
    fig.tight_layout()
    return jsonify({'img': fig_to_b64(fig)})

if __name__ == '__main__':
    app.run(debug=True)