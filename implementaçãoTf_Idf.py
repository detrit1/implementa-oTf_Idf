import math
import string
from collections import Counter
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False
    print("[aviso] reportlab não instalado. Execute: pip install reportlab")
    print("[aviso] O relatório PDF não será gerado.\n")

# Corpus
documents = [
    "O gato preto subiu no telhado para caçar aves.",
    "Cachorros grandes precisam de muito espaço para correr.",
    "O gato e o cachorro são animais domésticos populares.",
    "Aviões grandes decolam rapidamente do aeroporto.",
    "O cachorro preto correu atrás do gato pelo jardim."
]

query = "gato preto"

# Pré-processamento
stopwords = {"o", "e", "de", "do", "da", "para", "no", "pelo", "são"}

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]
    return tokens

processed_docs  = [preprocess(doc) for doc in documents]
processed_query = preprocess(query)

# Vocabulário
vocab = sorted(set(word for doc in processed_docs for word in doc))

# TF (Term Frequency)
def compute_tf(doc):
    tf = Counter(doc)
    return {word: tf.get(word, 0) for word in vocab}

tf_docs = [compute_tf(doc) for doc in processed_docs]

# IDF (Inverse Document Frequency)
N = len(documents)

def compute_idf():
    idf = {}
    for word in vocab:
        df = sum(1 for doc in processed_docs if word in doc)
        idf[word] = math.log(N / (1 + df))  # +1 para evitar divisão por zero
    return idf

idf = compute_idf()

# TF-IDF
def compute_tfidf(tf):
    return {word: tf[word] * idf[word] for word in vocab}

tfidf_docs = [compute_tfidf(tf) for tf in tf_docs]

# Vetor da Query
query_tf    = compute_tf(processed_query)
query_tfidf = compute_tfidf(query_tf)

# Similaridade de Cosseno
def cosine_similarity(vec1, vec2):
    dot   = sum(vec1[w] * vec2[w] for w in vocab)
    norm1 = math.sqrt(sum(vec1[w]**2 for w in vocab))
    norm2 = math.sqrt(sum(vec2[w]**2 for w in vocab))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

# Ranking
scores = []
for i, doc_vec in enumerate(tfidf_docs):
    score = cosine_similarity(query_tfidf, doc_vec)
    scores.append((i, score))

scores.sort(key=lambda x: x[1], reverse=True)

# Resultado no terminal (comportamento original)
print("Ranking dos documentos:\n")
for idx, score in scores:
    print(f"Doc {idx+1} | Score: {score:.4f}")
    print(documents[idx])
    print()

# Geração do relatório PDF
def gerar_relatorio_pdf(output_path="relatorio_tfidf.pdf"):
    if not REPORTLAB_OK:
        return

    W = A4[0] - 4*cm

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()

    title_s = ParagraphStyle("titulo", parent=styles["Title"],
        fontSize=18, textColor=colors.HexColor("#1a1a2e"), spaceAfter=4)
    h1_s = ParagraphStyle("h1", parent=styles["Heading1"],
        fontSize=13, textColor=colors.HexColor("#16213e"),
        spaceBefore=14, spaceAfter=4)
    h2_s = ParagraphStyle("h2", parent=styles["Heading2"],
        fontSize=11, textColor=colors.HexColor("#0f3460"),
        spaceBefore=6, spaceAfter=2)
    body_s = ParagraphStyle("corpo", parent=styles["Normal"],
        fontSize=9, leading=14, textColor=colors.HexColor("#333333"))
    code_s = ParagraphStyle("code", parent=styles["Normal"],
        fontSize=8, leading=12, fontName="Courier",
        textColor=colors.HexColor("#1a1a2e"),
        backColor=colors.HexColor("#f0f0f5"),
        leftIndent=8, rightIndent=8, borderPad=6,
        spaceBefore=2, spaceAfter=2)
    sub_s = ParagraphStyle("sub", parent=body_s,
        fontSize=8, textColor=colors.HexColor("#555555"), spaceAfter=2)

    HDR  = colors.HexColor("#16213e")
    ROW1 = colors.HexColor("#e8eaf6")
    ROW2 = colors.white
    RANK_COLORS = [
        colors.HexColor("#c8e6c9"),
        colors.HexColor("#dcedc8"),
        colors.HexColor("#fff9c4"),
        colors.HexColor("#ffe0b2"),
        colors.HexColor("#ffcdd2"),
    ]

    def rule():
        return HRFlowable(width="100%", thickness=0.5,
                          color=colors.HexColor("#16213e"), spaceAfter=6)

    def make_table(data, col_widths):
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("FONTNAME",       (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",       (0,0), (-1,0),  9),
            ("BACKGROUND",     (0,0), (-1,0),  HDR),
            ("TEXTCOLOR",      (0,0), (-1,0),  colors.white),
            ("ALIGN",          (0,0), (-1,-1), "CENTER"),
            ("VALIGN",         (0,0), (-1,-1), "MIDDLE"),
            ("FONTSIZE",       (0,1), (-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [ROW1, ROW2]),
            ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#aaaaaa")),
            ("TOPPADDING",     (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",  (0,0), (-1,-1), 3),
        ]))
        return t

    sp = lambda n=0.3: Spacer(1, n*cm)
    story = []

    # Relatório

    # Capa
    story += [
        sp(1),
        Paragraph("Relatório: Busca por Similaridade TF-IDF", title_s),
        Paragraph(f'Query: <font color="#c0392b"><b>"{query}"</b></font>', h2_s),
        rule(), sp(),
    ]

    # Corpus
    story += [
        Paragraph("Etapa 1 — Corpus de Documentos", h1_s), rule(),
        Paragraph("Conjunto de documentos sobre o qual a busca será realizada.", body_s),
        sp(0.2),
    ]
    data = [["#", "Documento original"]]
    for i, d in enumerate(documents):
        data.append([f"Doc {i+1}", d])
    story += [make_table(data, [1.2*cm, W - 1.2*cm]), sp()]

    # Pré-processamento
    story += [
        Paragraph("Etapa 2 — Pré-processamento", h1_s), rule(),
        Paragraph(
            "Minúsculas → remove pontuação → remove stopwords. "
            f'Stopwords: <i>{", ".join(sorted(stopwords))}</i>.', body_s),
        sp(0.2),
    ]
    data = [["#", "Tokens após pré-processamento"]]
    for i, tokens in enumerate(processed_docs):
        data.append([f"Doc {i+1}", "  |  ".join(tokens)])
    data.append(["Query", "  |  ".join(processed_query)])
    story += [make_table(data, [1.2*cm, W - 1.2*cm]), sp()]

    # Vocabulário
    story += [
        Paragraph("Etapa 3 — Vocabulário", h1_s), rule(),
        Paragraph(f"União de todas as palavras únicas. Total: <b>{len(vocab)} palavras</b>.", body_s),
        sp(0.2),
        Paragraph("  ".join(vocab), code_s),
        sp(),
    ]

    # TF 
    story += [
        Paragraph("Etapa 4 — Term Frequency (TF)", h1_s), rule(),
        Paragraph("Contagem bruta de cada termo do vocabulário em cada documento.", body_s),
        sp(0.2)
    ]
    nonzero = [w for w in vocab if any(tf[w] > 0 for tf in tf_docs)]
    cw = min(3*cm, (W - 1.2*cm) / len(nonzero))

    header_style = ParagraphStyle(
        "header", fontName="Helvetica-Bold", fontSize=5,
        textColor=colors.white, leading=6,
        wordWrap='CJK',
    )

    data = [[Paragraph("Doc", header_style)] + [Paragraph(w, header_style) for w in nonzero]]
    for i, tf in enumerate(tf_docs):
        data.append([f"Doc {i+1}"] + [str(tf[w]) for w in nonzero])
    data.append(["Query"] + [str(query_tf[w]) for w in nonzero])

    t = Table(data, colWidths=[1.2*cm] + [cw]*len(nonzero))
    t.setStyle(TableStyle([
        ("FONTNAME",       (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0), (-1,-1), 7),
        ("FONTSIZE",       (0,0), (-1,0),  5),
        ("BACKGROUND",     (0,0), (-1,0),  HDR),
        ("TEXTCOLOR",      (0,0), (-1,0),  colors.white),
        ("ALIGN",          (0,0), (-1,-1), "CENTER"),
        ("VALIGN",         (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [ROW1, ROW2]),
        ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#aaaaaa")),
        ("TOPPADDING",     (0,0), (-1,-1), 2),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 2),
    ]))
    story += [t, sp()]

    # IDF 
    story += [
        Paragraph("Etapa 5 — Inverse Document Frequency (IDF)", h1_s), rule(),
        Paragraph(
            "IDF(t) = log( N / (1 + df) ). "
            "Palavras raras recebem peso maior; palavras comuns recebem peso menor.", body_s),
        sp(0.2),
    ]
    data = [["Palavra", "Doc. freq. (df)", "IDF = log(5/(1+df))"]]
    for w in vocab:
        df = sum(1 for d in processed_docs if w in d)
        data.append([w, str(df), f"{idf[w]:.4f}"])
    story += [make_table(data, [4*cm, 4*cm, W - 8*cm]), sp()]

    # TF-IDF
    story += [
        Paragraph("Etapa 6 — TF-IDF", h1_s), rule(),
        Paragraph("TF-IDF(t, d) = TF(t, d) × IDF(t). Termos com zero omitidos.", body_s),
        sp(0.2),
    ]
    cw2 = min(1.6*cm, (W - 1.2*cm) / len(nonzero))

    header_style = ParagraphStyle(
        "header", fontName="Helvetica-Bold", fontSize=5,
        textColor=colors.white, leading=6,
        wordWrap='CJK',
    )

    data = [[Paragraph("Doc", header_style)] + [Paragraph(w, header_style) for w in nonzero]]
    for i, tfidf in enumerate(tfidf_docs):
        data.append([f"Doc {i+1}"] + [f"{tfidf[w]:.3f}" for w in nonzero])
    data.append(["Query"] + [f"{query_tfidf[w]:.3f}" for w in nonzero])

    t = Table(data, colWidths=[1.2*cm] + [cw2]*len(nonzero))
    t.setStyle(TableStyle([
        ("FONTNAME",       (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0), (-1,-1), 7),
        ("FONTSIZE",       (0,0), (-1,0),  5),
        ("BACKGROUND",     (0,0), (-1,0),  HDR),
        ("TEXTCOLOR",      (0,0), (-1,0),  colors.white),
        ("ALIGN",          (0,0), (-1,-1), "CENTER"),
        ("VALIGN",         (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [ROW1, ROW2]),
        ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#aaaaaa")),
        ("TOPPADDING",     (0,0), (-1,-1), 2),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 2),
    ]))
    story += [t, sp()]

    # Similaridade de Cosseno
    story += [
        Paragraph("Etapa 7 — Similaridade de Cosseno", h1_s), rule(),
        Paragraph(
            "Score = (vec_doc · vec_query) / (|vec_doc| × |vec_query|). "
            "Resultado entre 0 (sem relação) e 1 (idênticos).", body_s),
        sp(0.2),
    ]
    data = [["Doc", "Dot product", "|doc|", "|query|", "Score (cos)"]]
    for i, doc_vec in enumerate(tfidf_docs):
        dot   = sum(doc_vec[w] * query_tfidf[w] for w in vocab)
        norm1 = math.sqrt(sum(doc_vec[w]**2 for w in vocab))
        norm2 = math.sqrt(sum(query_tfidf[w]**2 for w in vocab))
        sc    = dot / (norm1 * norm2) if norm1 and norm2 else 0
        data.append([f"Doc {i+1}", f"{dot:.4f}", f"{norm1:.4f}", f"{norm2:.4f}", f"{sc:.4f}"])
    story += [make_table(data, [1.2*cm, 3*cm, 2.5*cm, 2.5*cm, W - 9.2*cm]), sp()]

    # Ranking Final
    story += [
        Paragraph("Etapa 8 — Ranking Final", h1_s), rule(),
        Paragraph(
            f'Documentos ordenados por relevância para a query <b>"{query}"</b>.', body_s),
        sp(0.2),
    ]
    data = [["Posição", "Doc", "Score", "Documento"]]
    for pos, (idx, score) in enumerate(scores):
        data.append([f"{pos+1}º", f"Doc {idx+1}", f"{score:.4f}", documents[idx]])
    t = Table(data, colWidths=[1.4*cm, 1.4*cm, 1.8*cm, W - 4.6*cm])
    cmds = [
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0),  9),
        ("BACKGROUND",    (0,0), (-1,0),  HDR),
        ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("ALIGN",         (3,1), (3,-1),  "LEFT"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("FONTSIZE",      (0,1), (-1,-1), 8),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#aaaaaa")),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]
    for pos in range(len(scores)):
        cmds.append(("BACKGROUND", (0, pos+1), (-1, pos+1), RANK_COLORS[pos]))
    t.setStyle(TableStyle(cmds))
    story += [t, sp(0.5), rule()]
    story.append(Paragraph(
        "Link para o repositório do GitHub: https://github.com/detrit1/implementa-oTf_Idf",
        sub_s))

    doc.build(story)
    print(f"\nRelatório PDF gerado: {output_path}")


gerar_relatorio_pdf("relatorio_tfidf.pdf")