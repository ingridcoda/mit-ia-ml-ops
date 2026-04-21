"""clustering.py — Agrupamento de Perfis de Crédito."""
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run(df, config, dirs, logger):
    logger.info("=== Clusterização e PCA de Perfis ===")

    # Seleciona numéricas para clusterização
    features = ['Age', 'Credit amount', 'Duration']
    X = df[features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. KMeans (3 perfis de clientes)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster_Perfil'] = kmeans.fit_predict(X_scaled)

    # 2. PCA (Redução para 2D para visualização)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df['pca_1'] = pca_result[:, 0]
    df['pca_2'] = pca_result[:, 1]

    # Salva variância explicada (Item 4 da Rubrica)
    var_exp = {"explained_variance_ratio": pca.explained_variance_ratio_.tolist()}
    with open(dirs['stats'] / "27_pca_variance.json", "w") as f:
        json.dump(var_exp, f, indent=4)

    return {"status": "success"}
