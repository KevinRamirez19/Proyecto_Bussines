import pandas as pd
import numpy as np

# ── Cargar datos ───────────────────────────────────────────────────────────────
df = pd.read_csv('../data/raw/googleplaystore.csv', quotechar='"', encoding='utf-8', on_bad_lines='skip')

# ── 1. Eliminar fila corrupta (Type = '0', datos desplazados) ─────────────────
df = df[df['Type'].isin(['Free', 'Paid'])].copy()

# ── 2. Eliminar duplicados por nombre de app ──────────────────────────────────
df = df.drop_duplicates(subset='App').reset_index(drop=True)

# ── 3. Rating → float, máximo 5.0 ────────────────────────────────────────────
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df['Rating'] = df['Rating'].clip(upper=5.0)

# ── 4. Reviews → entero (maneja casos como '3.0M') ───────────────────────────
df['Reviews'] = df['Reviews'].str.replace('M', '000000', regex=False)
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce').astype('Int64')

# ── 5. Size → Size_MB en float ───────────────────────────────────────────────
def parse_size(s):
    s = str(s).strip()
    if s.endswith('M'):
        return float(s[:-1])
    elif s.endswith('k'):
        return round(float(s[:-1]) / 1024, 4)
    else:
        return np.nan  # 'Varies with device' → NaN

df['Size_MB'] = df['Size'].apply(parse_size)
df = df.drop(columns=['Size'])

# ── 6. Installs → entero (quita '+' y ',') ───────────────────────────────────
df['Installs'] = (
    df['Installs']
    .str.replace('+', '', regex=False)
    .str.replace(',', '', regex=False)
)
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce').astype('Int64')

# ── 7. Type → bool (False=Free, True=Paid) ───────────────────────────────────
df['Type'] = df['Type'].map({'Free': False, 'Paid': True})

# ── 8. Price → float (quita el símbolo '$') ──────────────────────────────────
df['Price'] = df['Price'].str.replace('$', '', regex=False)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0.0)

# ── 9. Last Updated → datetime ───────────────────────────────────────────────
df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce').dt.date

# ── Verificar resultado ───────────────────────────────────────────────────────
print("Shape final:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nNulos restantes:\n", df.isnull().sum())
print("\nMuestra:\n", df.head(3))

# ── Exportar como .xlsx ───────────────────────────────────────────────────────
df.to_excel('../data/processed/googleplaystore_limpieza.xlsx', index=False)  
print("Archivo guardado: googleplaystore_limpieza.xlsx")