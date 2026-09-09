import streamlit as st
import os
import json
import time
import subprocess
import tempfile
from pathlib import Path

import requests
import imageio_ffmpeg
from groq import Groq

# Caminho do binário do ffmpeg, incluído diretamente no pacote pip
# "imageio-ffmpeg" (não depende de apt-get nem de downloads em tempo de
# execução, o que evita os problemas de infraestrutura vistos em hospedagens
# gratuitas como o Streamlit Cloud).
CAMINHO_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# ----------------------------------------------------------------------------
# CONFIGURAÇÃO GERAL DA PÁGINA
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Transcritor Jurídico",
    page_icon="⚖️",
    layout="centered",
)

# Modelo usado na Groq apenas para o resumo executivo (texto -> texto)
MODELO_RESUMO = "openai/gpt-oss-120b"  # modelo de chat gratuito na Groq

# Endpoints da AssemblyAI (transcrição + separação de falantes/diarização)
ASSEMBLYAI_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
ASSEMBLYAI_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"

EXTENSOES_PERMITIDAS = ["mp4", "mp3", "wav", "m4a", "mov"]

# ----------------------------------------------------------------------------
# ESTADO DA SESSÃO
# ----------------------------------------------------------------------------
if "transcricao" not in st.session_state:
    st.session_state.transcricao = ""
if "resumo" not in st.session_state:
    st.session_state.resumo = ""


# ----------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ----------------------------------------------------------------------------
def get_cliente_groq():
    """Cria o cliente da Groq usando a chave informada pelo usuário (para o resumo)."""
    api_key = st.session_state.get("groq_api_key", "").strip()
    if not api_key:
        st.error("⚠️ Informe sua chave de API da Groq na barra lateral antes de continuar.")
        st.stop()
    return Groq(api_key=api_key)


def get_chave_assemblyai() -> str:
    """Recupera a chave da AssemblyAI informada pelo usuário (para a transcrição)."""
    api_key = st.session_state.get("assemblyai_api_key", "").strip()
    if not api_key:
        st.error("⚠️ Informe sua chave de API da AssemblyAI na barra lateral antes de continuar.")
        st.stop()
    return api_key


def preparar_audio(arquivo_carregado) -> str:
    """
    Recebe o arquivo enviado pelo usuário (vídeo ou áudio), extrai/otimiza o
    áudio chamando o ffmpeg diretamente (sem pydub) e salva como um .mp3
    temporário, mono e com taxa de amostragem reduzida — isso diminui o
    tamanho do arquivo antes de enviar para a API e garante compatibilidade
    mesmo com vídeos (.mp4, .mov).
    """
    sufixo_original = Path(arquivo_carregado.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=sufixo_original) as tmp_in:
        tmp_in.write(arquivo_carregado.getbuffer())
        caminho_entrada = tmp_in.name

    caminho_saida = caminho_entrada + "_convertido.mp3"

    comando = [
        CAMINHO_FFMPEG,
        "-y",              # sobrescreve o arquivo de saída se já existir
        "-i", caminho_entrada,
        "-vn",             # descarta qualquer trilha de vídeo/imagem
        "-ac", "1",        # áudio mono
        "-ar", "16000",    # taxa de amostragem reduzida
        "-b:a", "64k",     # bitrate baixo (arquivo final bem mais leve)
        caminho_saida,
    ]
    resultado = subprocess.run(
        comando, capture_output=True, text=True
    )

    os.remove(caminho_entrada)

    if resultado.returncode != 0 or not os.path.exists(caminho_saida):
        raise RuntimeError(
            "Falha ao converter o arquivo de áudio/vídeo com o ffmpeg:\n"
            f"{resultado.stderr[-800:]}"
        )

    return caminho_saida


def transcrever_com_diarizacao(caminho_mp3: str, idioma: str, barra_progresso=None) -> str:
    """
    Envia o áudio para a AssemblyAI, que faz a transcrição E identifica
    automaticamente os diferentes falantes (diarização). Retorna o texto já
    formatado como 'Falante A: ...', 'Falante B: ...' etc.
    """
    chave = get_chave_assemblyai()
    headers = {"authorization": chave}

    # 1) Envia o arquivo de áudio para a AssemblyAI
    with open(caminho_mp3, "rb") as arquivo_audio:
        resposta_upload = requests.post(
            ASSEMBLYAI_UPLOAD_URL, headers=headers, data=arquivo_audio
        )
    resposta_upload.raise_for_status()
    url_audio = resposta_upload.json()["upload_url"]

    # 2) Solicita a transcrição com identificação de falantes habilitada
    corpo_requisicao = {
        "audio_url": url_audio,
        "speaker_labels": True,   # habilita a separação de vozes
        "punctuate": True,
        "format_text": True,
    }
    if idioma == "auto":
        corpo_requisicao["language_detection"] = True
    else:
        corpo_requisicao["language_code"] = idioma

    resposta_criacao = requests.post(
        ASSEMBLYAI_TRANSCRIPT_URL, headers=headers, json=corpo_requisicao
    )
    resposta_criacao.raise_for_status()
    id_transcricao = resposta_criacao.json()["id"]

    # 3) Aguarda o processamento (a AssemblyAI processa de forma assíncrona)
    url_status = f"{ASSEMBLYAI_TRANSCRIPT_URL}/{id_transcricao}"
    tempo_maximo_espera = 600  # 10 minutos de segurança
    tempo_decorrido = 0

    while True:
        resposta_status = requests.get(url_status, headers=headers)
        resposta_status.raise_for_status()
        dados = resposta_status.json()
        status = dados["status"]

        if status == "completed":
            break
        if status == "error":
            raise RuntimeError(f"A AssemblyAI retornou um erro: {dados.get('error')}")
        if tempo_decorrido >= tempo_maximo_espera:
            raise TimeoutError("A transcrição demorou demais e foi interrompida.")

        time.sleep(3)
        tempo_decorrido += 3
        if barra_progresso is not None:
            progresso = min(0.95, tempo_decorrido / 90)  # estimativa visual
            barra_progresso.progress(progresso)

    if barra_progresso is not None:
        barra_progresso.progress(1.0)

    # 4) Monta o texto final já separado por falante
    utterances = dados.get("utterances")
    if utterances:
        linhas = [f"Falante {u['speaker']}: {u['text']}" for u in utterances]
        return "\n\n".join(linhas)

    # Se por algum motivo a diarização não retornar 'utterances', usa o texto puro
    return dados.get("text", "")


def gerar_resumo_juridico(texto_transcrito: str) -> str:
    """Usa um modelo de linguagem da Groq para gerar um resumo executivo jurídico."""
    cliente = get_cliente_groq()

    prompt_sistema = (
        "Você é um assistente jurídico especializado em analisar transcrições de "
        "atendimentos e reuniões de advocacia, já identificadas por falante (Falante A, "
        "Falante B etc). A partir do texto transcrito, produza um RESUMO EXECUTIVO claro "
        "e objetivo, em português, organizado exatamente nas seguintes seções:\n\n"
        "1. PARTES ENVOLVIDAS - relacione os falantes identificados (Falante A, B...) com "
        "o papel que parecem ocupar na conversa, quando isso for dedutível do conteúdo.\n"
        "2. FATOS PRINCIPAIS - descreva em tópicos os fatos relevantes narrados.\n"
        "3. DATAS E PRAZOS MENCIONADOS - liste datas, prazos ou períodos citados.\n"
        "4. PONTOS DE ATENÇÃO / PRÓXIMOS PASSOS - questões que exigem providência do advogado.\n\n"
        "Se alguma seção não tiver informação disponível no texto, escreva "
        "'Não identificado na transcrição'. Seja fiel apenas ao que está no texto, "
        "sem inventar informações."
    )

    resposta = cliente.chat.completions.create(
        model=MODELO_RESUMO,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": texto_transcrito},
        ],
        temperature=0.2,
    )
    return resposta.choices[0].message.content


def botao_copiar(texto: str, rotulo: str, chave: str):
    """Cria um botão HTML/JS que copia o texto para a área de transferência."""
    texto_js = json.dumps(texto)
    html = f"""
    <button onclick='navigator.clipboard.writeText({texto_js})'
        style="
            background-color:#f0f2f6;
            border:1px solid #d0d2d6;
            border-radius:8px;
            padding:8px 16px;
            cursor:pointer;
            font-size:14px;
            width:100%;
        ">
        📋 {rotulo}
    </button>
    """
    st.components.v1.html(html, height=45)


# ----------------------------------------------------------------------------
# BARRA LATERAL
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("🔑 Configuração")

    st.text_input(
        "Chave de API da AssemblyAI",
        type="password",
        key="assemblyai_api_key",
        help="Crie gratuitamente em https://www.assemblyai.com/app/api-keys — usada para transcrever e separar os falantes.",
    )
    st.text_input(
        "Chave de API da Groq",
        type="password",
        key="groq_api_key",
        help="Crie gratuitamente em https://console.groq.com/keys — usada só para gerar o resumo executivo.",
    )
    st.caption(
        "Suas chaves não são salvas em nenhum lugar — ficam apenas nesta sessão do navegador."
    )

    st.divider()
    idioma = st.selectbox(
        "Idioma do áudio",
        options=["pt", "auto"],
        format_func=lambda x: "Português" if x == "pt" else "Detectar automaticamente",
    )

    st.divider()
    st.caption(
        "Como conseguir as chaves gratuitas:\n\n"
        "**AssemblyAI** (transcrição + separação de falantes):\n"
        "1. Acesse assemblyai.com e crie uma conta grátis\n"
        "2. Vá em 'API Keys' e copie sua chave\n\n"
        "**Groq** (resumo executivo):\n"
        "1. Acesse console.groq.com\n"
        "2. Crie uma conta gratuita\n"
        "3. Vá em 'API Keys' e clique em 'Create API Key'"
    )

# ----------------------------------------------------------------------------
# CABEÇALHO
# ----------------------------------------------------------------------------
st.title("⚖️ Transcritor de Atendimentos Jurídicos")
st.write(
    "Envie o áudio ou vídeo de um atendimento ou reunião para transcrever "
    "automaticamente com identificação de cada falante e, se quiser, gerar "
    "um resumo executivo dos pontos principais."
)

# ----------------------------------------------------------------------------
# UPLOAD DE ARQUIVO
# ----------------------------------------------------------------------------
arquivo = st.file_uploader(
    "Selecione o arquivo de áudio ou vídeo",
    type=EXTENSOES_PERMITIDAS,
    help="Formatos aceitos: MP4, MP3, WAV, M4A, MOV",
)

if arquivo is not None:
    tamanho_mb = arquivo.size / (1024 * 1024)
    st.caption(f"Arquivo selecionado: **{arquivo.name}** ({tamanho_mb:.1f} MB)")

col1, col2 = st.columns(2)

with col1:
    iniciar = st.button("🎙️ Iniciar Transcrição", type="primary", use_container_width=True)

with col2:
    limpar = st.button("🗑️ Limpar Resultados", use_container_width=True)

if limpar:
    st.session_state.transcricao = ""
    st.session_state.resumo = ""
    st.rerun()

# ----------------------------------------------------------------------------
# PROCESSO DE TRANSCRIÇÃO
# ----------------------------------------------------------------------------
if iniciar:
    if arquivo is None:
        st.warning("Por favor, envie um arquivo antes de iniciar a transcrição.")
    else:
        try:
            with st.spinner("Preparando o áudio..."):
                caminho_mp3 = preparar_audio(arquivo)

            st.info("🎙️ Transcrevendo e identificando os falantes... isso pode levar de 1 a 3 minutos.")
            barra = st.progress(0.0)
            texto = transcrever_com_diarizacao(caminho_mp3, idioma, barra_progresso=barra)
            st.session_state.transcricao = texto.strip()
            st.session_state.resumo = ""  # reseta resumo anterior

            os.remove(caminho_mp3)
            st.success("Transcrição concluída com sucesso!")
        except Exception as erro:
            st.error(f"Ocorreu um erro durante a transcrição: {erro}")

# ----------------------------------------------------------------------------
# EXIBIÇÃO DA TRANSCRIÇÃO
# ----------------------------------------------------------------------------
if st.session_state.transcricao:
    st.subheader("📝 Transcrição (por falante)")
    st.text_area(
        "Resultado da transcrição",
        value=st.session_state.transcricao,
        height=300,
        key="caixa_transcricao",
        label_visibility="collapsed",
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        botao_copiar(st.session_state.transcricao, "Copiar Texto", "copiar_transcricao")

    with col_b:
        st.download_button(
            label="⬇️ Baixar (.txt)",
            data=st.session_state.transcricao,
            file_name="transcricao.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col_c:
        gerar_resumo_btn = st.button(
            "📋 Gerar Resumo Executivo", use_container_width=True
        )

    if gerar_resumo_btn:
        try:
            with st.spinner("Analisando a transcrição e gerando o resumo..."):
                st.session_state.resumo = gerar_resumo_juridico(
                    st.session_state.transcricao
                )
        except Exception as erro:
            st.error(f"Ocorreu um erro ao gerar o resumo: {erro}")

# ----------------------------------------------------------------------------
# EXIBIÇÃO DO RESUMO EXECUTIVO
# ----------------------------------------------------------------------------
if st.session_state.resumo:
    st.subheader("📌 Resumo Executivo")
    st.text_area(
        "Resumo executivo",
        value=st.session_state.resumo,
        height=300,
        key="caixa_resumo",
        label_visibility="collapsed",
    )

    col_d, col_e = st.columns(2)
    with col_d:
        botao_copiar(st.session_state.resumo, "Copiar Resumo", "copiar_resumo")
    with col_e:
        st.download_button(
            label="⬇️ Baixar Resumo (.txt)",
            data=st.session_state.resumo,
            file_name="resumo_executivo.txt",
            mime="text/plain",
            use_container_width=True,
        )

st.divider()
st.caption(
    "Este aplicativo utiliza a API gratuita da AssemblyAI (transcrição + identificação "
    "de falantes) e da Groq (resumo executivo), e não armazena nenhum arquivo ou dado "
    "após o fechamento da sessão."
)
