import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pymongo import MongoClient
import gridfs
import io

def app_visualizar_imagens():
    st.title("🔍 Sistema de Similaridade de Faces – MongoDB + Streamlit")

    # -----------------------------
    # 1. CONEXÃO AO MONGODB
    # -----------------------------
    URI = "mongodb+srv://joaokvicente_db_user:yuwBg4rA4ERff29G@cluster0.rpj5551.mongodb.net/?appName=Cluster0"
    DB_NAME = 'midias'

    client = MongoClient(URI)
    db = client[DB_NAME]
    fs = gridfs.GridFS(db)
    st.info("Conectando ao banco...")

    # -----------------------------
    # 2. CARREGAR IMAGENS DO GRIDFS
    # -----------------------------
    arquivos = list(fs.find())
    if not arquivos:
        st.error("Nenhuma imagem encontrada no banco.")
        return

    dataset = []
    nomes = []

    for arquivo in arquivos:
        dados = arquivo.read()
        img = Image.open(io.BytesIO(dados)).convert("L")
        img = img.resize((100, 100))
        dataset.append(np.array(img))
        nomes.append(arquivo.filename)

    dataset = np.array(dataset)
    st.success(f"{len(dataset)} imagens carregadas do banco.")

    # -----------------------------
    # 3. ESCOLHER MÉTODO DE ENTRADA
    # -----------------------------
    st.subheader("📸 Enviar imagem:")
    opcao = st.radio("Escolha um método:", ["Tirar foto com a câmera", "Upload de imagem"])
    img_user = None

    if opcao == "Tirar foto com a câmera":
        foto = st.camera_input("Tire uma foto agora:")
        if foto is not None:
            img_user = Image.open(foto).convert("L")
    elif opcao == "Upload de imagem":
        file = st.file_uploader("Selecione uma imagem:", type=["jpg", "jpeg", "png"])
        if file is not None:
            img_user = Image.open(file).convert("L")

    if img_user is None:
        st.warning("Envie uma imagem ou tire uma foto para continuar.")
        return

    # -----------------------------
    # 4. PREPARAR IMAGEM DO USUÁRIO
    # -----------------------------
    img_user = img_user.resize((100, 100))
    foto_array = np.array(img_user)
    st.image(img_user, caption="📥 Imagem recebida", width=200)

    # -----------------------------
    # 5. CALCULAR SIMILARIDADE
    # -----------------------------
    diffs = np.zeros(len(dataset))
    for i in range(len(dataset)):
        diffs[i] = np.sum(abs(dataset[i] - foto_array))

    id_mais = np.argmin(diffs)
    id_menos = np.argmax(diffs)
    img_mais = dataset[id_mais]
    img_menos = dataset[id_menos]

    # -----------------------------
    # 6. EXIBIR RESULTADOS
    # -----------------------------
    st.subheader("Resultados da análise")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_mais, caption=f"✔ Mais parecida ({nomes[id_mais]})", width=200)
    with col2:
        st.image(img_menos, caption=f"❌ Menos parecida ({nomes[id_menos]})", width=200)

    # -----------------------------
    # 7. GRÁFICO COMPARATIVO
    # -----------------------------
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].imshow(foto_array, cmap="gray")
    ax[0].set_title("Imagem enviada")
    ax[0].axis("off")
    ax[1].imshow(img_mais, cmap="gray")
    ax[1].set_title("Mais parecida")
    ax[1].axis("off")
    ax[2].imshow(img_menos, cmap="gray")
    ax[2].set_title("Menos parecida")
    ax[2].axis("off")
    st.pyplot(fig)


if __name__ == '__main__':
    app_visualizar_imagens()
