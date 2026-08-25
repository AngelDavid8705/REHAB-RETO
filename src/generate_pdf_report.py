"""
src/generate_pdf_report.py
Generador automatizado del Reporte PDF Ejecutivo de EDA para el dataset REHAB (Rehab_exercise).
Sigue rigurosamente la estructura de 3 pasos de Miriam Santos (Towards Data Science)
y las mejores prácticas de visualización y redacción clínica/técnica para un Senior Data Analyst (+10 años).
Diseño visual impecable de 9 páginas perfectamente balanceadas sin desbordes.
"""

import os
import sys
import json
from pathlib import Path

# Agregar raíz del proyecto a sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """
    Canvas personalizado de 2 pasadas para agregar encabezados ejecutivos
    y numeración dinámica de páginas 'Página X de Y'.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # En la primera página (portada) omitimos el encabezado y pie superior
        if self._pageNumber > 1:
            # Encabezado
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#1A365D"))
            self.drawString(45, 752, "REPORTE DE ANÁLISIS EXPLORATORIO DE DATOS (EDA) | DATASET REHAB (TRAIN SET)")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#718096"))
            self.drawRightString(612 - 45, 752, "Metodología CRISP-DM • IA Avanzada")
            
            # Línea divisoria superior
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.75)
            self.line(45, 746, 612 - 45, 746)
            
            # Pie de página
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.75)
            self.line(45, 40, 612 - 45, 40)
            
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#718096"))
            self.drawString(45, 28, "Confidencial • Reporte Técnico Senior Data Analyst • REHAB-RETO")
            page_text = f"Página {self._pageNumber} de {page_count}"
            self.drawRightString(612 - 45, 28, page_text)
            
        self.restoreState()


def build_pdf_report(json_path="reports/eda_results.json", output_pdf="reports/Reporte_EDA_REHAB_Ejercicios_Entrenamiento.pdf"):
    with open(json_path, "r") as f:
        eda_data = json.load(f)
        
    summary = eda_data["summary"]
    channels = eda_data["channels"]
    classes = eda_data["classes"]
    figures = eda_data["figures"]
    
    # Márgenes calibrados: 45pt izquierda/derecha, 45pt arriba/abajo
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=letter,
        leftMargin=45,
        rightMargin=45,
        topMargin=45,
        bottomMargin=45
    )
    
    styles = getSampleStyleSheet()
    
    # Paleta cromática ejecutiva y temática médica/datos
    c_primary = colors.HexColor("#1A365D")      # Azul marino profundo
    c_secondary = colors.HexColor("#2B6CB0")    # Azul profesional
    c_dark = colors.HexColor("#2D3748")         # Gris carbón texto
    c_accent = colors.HexColor("#D69E2E")       # Dorado ámbar
    c_alert = colors.HexColor("#C53030")        # Carmesí alerta
    c_success = colors.HexColor("#2F855A")      # Verde esmeralda
    c_light_bg = colors.HexColor("#F7FAFC")     # Fondo claro tabla
    c_card_bg = colors.HexColor("#EDF2F7")      # Fondo tarjeta metadata
    c_border = colors.HexColor("#CBD5E0")       # Borde sutil
    
    # Estilos tipográficos
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_primary,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=c_secondary,
        spaceAfter=10
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16,
        textColor=c_primary,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13.5,
        textColor=c_secondary,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=c_dark,
        spaceAfter=4
    )
    
    body_bold = ParagraphStyle(
        'BodyBold_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=12,
        textColor=c_dark,
        spaceAfter=4
    )
    
    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#2C5282")
    )
    
    alert_style = ParagraphStyle(
        'AlertText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#742A2A")
    )
    
    caption_style = ParagraphStyle(
        'CaptionStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,  # Centrado
        spaceBefore=3,
        spaceAfter=6
    )
    
    table_text = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        leading=9.5,
        textColor=c_dark
    )
    
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    )
    
    story = []
    
    # =========================================================
    # PÁGINA 1: PORTADA, RESUMEN EJECUTIVO & CONTEXTO BIOMÉDICO
    # =========================================================
    story.append(Paragraph("REPORTE INTEGRAL DE ANÁLISIS EXPLORATORIO DE DATOS (EDA)", title_style))
    story.append(Paragraph("<b>Diagnóstico Cinemático Multimodal, Evaluación Forense de Calidad de Datos y Guía Metodológica para la Clasificación de Tareas de Rehabilitación Post-ACV (Dataset REHAB)</b>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=c_primary, spaceBefore=0, spaceAfter=8))
    
    meta_html = f"""
    <b>Proyecto:</b> Reto de Clasificación de Movimientos de Rehabilitación (REHAB-RETO) | <b>Metodología:</b> CRISP-DM<br/>
    <b>Autor:</b> Lead Data Scientist & Senior Kinematics Analyst (+10 años exp) | <b>Equipo:</b> Angel Lugo, Jose Pablo, Juan Pablo<br/>
    <b>Dataset:</b> REHAB - <i>Rehab_exercise</i> (Lv et al., <i>Nature Scientific Data</i> 2026) | <b>Scope:</b> Set de Entrenamiento (N=2,979 ensayos, 70% split estratificado)
    """
    meta_table = Table([[Paragraph(meta_html, body_style)]], colWidths=[522])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_card_bg),
        ('BOX', (0, 0), (-1, -1), 1, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 6))
    
    exec_summary_text = """
    <b>DICTAMEN EJECUTIVO DEL SENIOR ANALYST:</b><br/>
    Este informe proporciona una auditoría integral, rigurosa y matemática del conjunto de datos de entrenamiento del proyecto <b>REHAB</b>. Siguiendo el estándar de 3 pasos para EDA de Miriam Santos (<i>Towards Data Science</i>), hemos auditado <b>31,458,240 registros numéricos cinemáticos</b> (2,979 ensayos de 15 movimientos post-ACV). Hallazgos clave: (1) <b>Integridad Numérica Impecable</b> (0 NaNs, 0 Infs), (2) <b>Normalización Trial-Wise a Media Cero</b>, condicionando la energía RMS y varianza como discriminadores primarios, (3) <b>Razón de Desbalance Moderada</b> (1.82:1, oscilando entre 148 y 269 muestras por clase) adecuada para modelado directo con estratificación, y (4) <b>Autopsia Forense de Corrupción</b> en <code>014_1.npy</code> en el repositorio fuente (Science Data Bank) por decodificación UTF-8, fundamentando la exclusión del Movimiento 14 del benchmark de 12 canales.
    """
    exec_table = Table([[Paragraph(exec_summary_text, callout_style)]], colWidths=[522])
    exec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EBF8FF")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#90CDF4")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(exec_table)
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("1. Contexto Biomédico, Topología de Sensores y Formulación del Reto", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=c_secondary, spaceBefore=1, spaceAfter=4))
    
    p_contexto_topologia = """
    El accidente cerebrovascular (ACV) es la principal causa de discapacidad motora en adultos. La neuro-rehabilitación clínica busca restaurar la funcionalidad mediante la repetición sistemática de tareas biomecánicas estandarizadas. El dataset <b>REHAB</b> (Lv et al., 2026) captura trayectorias de <b>120 pacientes post-ACV</b> durante 2 semanas de terapia bajo el protocolo <i>Rehab_exercise</i> (16 ejercicios clínicos).
    <br/><br/>
    <b>Topología Multimodal de Sensado (12 Canales Cinemáticos Sincronizados a 880 Timesteps):</b>
    <br/>
    • <b>Sensor 1 (Unidad Inercial IMU - Canales 0 a 5):</b> Dos sensores IMU triaxiales que registran ángulos de Euler (°). En miembro superior se ubican en el brazo proximal (Canales 0-2: Pitch, Yaw, Roll) y antebrazo distal (Canales 3-5: Pitch, Yaw, Roll); en miembro inferior en muslo y pantorrilla.
    <br/>
    • <b>Sensor 2 (Guante de Flexión Sensorial + IMU de Muñeca - Canales 6 a 11):</b> Cinco sensores resistivos de flexión (Canales 6-10: Pulgar F1, Índice F2, Medio F3, Anular F4 y Meñique F5) en unidades arbitrarias continuas proporcionales al ángulo articular, más un sensor inercial en el dorso de la mano (Canal 11: Pitch S5 de muñeca en grados).
    <br/><br/>
    <b>Formulación Matemática del Problema:</b> Dado un tensor de entrada $\\mathbf{X}_i \\in \\mathbb{R}^{880 \\times 12}$ correspondiente a un ensayo cinemático $i$, el objetivo del sistema de Machine Learning es clasificar el movimiento ejecutado $y_i \\in \\mathcal{Y}$, donde $\\mathcal{Y} = \\{0, 1, \\dots, 15\\} \\setminus \\{14\\}$ comprende las 15 tareas terapéuticas válidas.
    """
    story.append(Paragraph(p_contexto_topologia, body_style))
    story.append(PageBreak())
    
    # =========================================================
    # PÁGINA 2: STEP 1 - DATASET OVERVIEW & DESCRIPTIVE STATS
    # =========================================================
    story.append(Paragraph("2. Paso 1: Visión General del Dataset y Estadística Descriptiva (Dataset Overview)", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=c_secondary, spaceBefore=1, spaceAfter=4))
    
    p_step1_text = f"""
    Siguiendo la metodología de Miriam Santos, el primer paso responde a: <i>¿Con qué datos estamos trabajando exactamente?</i>. Para evitar fuga de datos (data leakage), este análisis exploratorio se restringe estrictamente a la partición de <b>Entrenamiento (X_train, y_train)</b>, que abarca el 70% del dataset consolidado (2,979 ensayos).
    <br/><br/>
    <b>Estructura Numérica y Descriptores Globales:</b>
    <br/>
    • <b>Ensayos de Entrenamiento (N):</b> 2,979 | <b>Puntos Temporales (T):</b> 880 timesteps | <b>Canales (C):</b> 12 canales.
    <br/>
    • <b>Volumen Numérico Total:</b> 2,979 × 880 × 12 = <b>31,458,240 registros numéricos</b> (240.01 MB en RAM en float64).
    <br/>
    • <b>Integridad de Datos:</b> 0 valores nulos (NaN = 0), 0 valores infinitos (Inf = 0). Tasa de completitud: 100.0%.
    """
    story.append(Paragraph(p_step1_text, body_style))
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("<b>Tabla 1: Resumen Estadístico Descriptivo Exhaustivo por Canal (Set de Entrenamiento, N=2,979)</b>", body_bold))
    
    table_data = [[
        Paragraph("Canal / Sensor", table_header),
        Paragraph("Tipo Sensor", table_header),
        Paragraph("Media (μ)", table_header),
        Paragraph("Desv. Est. (σ)", table_header),
        Paragraph("Mediana", table_header),
        Paragraph("IQR", table_header),
        Paragraph("Mín / Máx", table_header),
        Paragraph("Asimetría", table_header),
        Paragraph("Curtosis", table_header),
        Paragraph("Outliers %", table_header),
    ]]
    
    for ch in channels:
        table_data.append([
            Paragraph(f"<b>C{ch['channel_idx']:02d}:</b> {ch['channel_name']}", table_text),
            Paragraph(ch['sensor_type'], table_text),
            Paragraph(f"{ch['mean']:.1e}", table_text),
            Paragraph(f"{ch['std']:.2f}", table_text),
            Paragraph(f"{ch['median']:.2f}", table_text),
            Paragraph(f"{ch['iqr']:.2f}", table_text),
            Paragraph(f"[{ch['min']:.0f}, {ch['max']:.0f}]", table_text),
            Paragraph(f"{ch['skewness']:.2f}", table_text),
            Paragraph(f"{ch['kurtosis']:.1f}", table_text),
            Paragraph(f"{ch['outliers_pct']:.1f}%", table_text),
        ])
        
    t_channels = Table(table_data, colWidths=[85, 55, 45, 45, 40, 40, 68, 44, 46, 44])
    t_channels.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 2),
        ('RIGHTPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(t_channels)
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("<b>Balance y Distribución de Clases Terapéuticas en el Set de Entrenamiento:</b>", h2_style))
    p_balance_desc = f"""
    En clasificación multiclase, el balance determina la propensión al sesgo algorítmico:
    <br/>
    • <b>Clase Mayoritaria:</b> Movimiento 07 (<i>Ball gripping</i>) con <b>269 muestras</b> (9.03% del total de entrenamiento).
    <br/>
    • <b>Clase Minoritaria:</b> Movimiento 01 (<i>Bobath flexion/extension</i>) con <b>148 muestras</b> (4.97% del total).
    <br/>
    • <b>Razón de Desbalance (Max / Min):</b> <b>1.82:1</b>.
    <br/>
    • <b>Entropía de Información de Shannon:</b> <b>2.70 nats</b> frente a un máximo teórico de $\\ln(15) = 2.708$ nats (<b>eficiencia informacional: 99.6%</b>).
    <br/>
    <b>Diagnóstico del Senior Analyst:</b> Esta distribución es <b>razonablemente balanceada</b> en el contexto biomédico. No se requieren técnicas agresivas de submuestreo o sobremuestreo sintético (SMOTE), pero es imperativo el uso de esquemas de <i>Stratified K-Fold</i> y métricas insensibles al desbalance como Macro F1-Score y Balanced Accuracy.
    """
    story.append(Paragraph(p_balance_desc, body_style))
    story.append(PageBreak())
    
    # =========================================================
    # PÁGINA 3: FIGURA 1 & STEP 2: ANÁLISIS UNIVARIADO
    # =========================================================
    story.append(Paragraph("3. Paso 2: Evaluación Profunda de Variables y Visualización (Feature Assessment)", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=c_secondary, spaceBefore=1, spaceAfter=4))
    
    if os.path.exists(figures["fig1"]):
        story.append(Image(figures["fig1"], width=6.8 * inch, height=2.6 * inch))
        story.append(Paragraph("Figura 1: Distribución y balance de muestras por clase en el conjunto de entrenamiento (N=2,979). Se resalta la clase mayoritaria (verde), minoritaria (rojo) y la línea media.", caption_style))
        story.append(Spacer(1, 4))
        
    story.append(Paragraph("3.1 Análisis Univariado: Morfología Probabilística, Asimetría y Dispersión por Canal", h2_style))
    p_univariate = """
    El análisis univariado examina la distribución de cada canal individual a través de sus 2.62 millones de puntos de tiempo:
    <br/>
    • <b>Propiedad de Media Cero (Mean-Centering Artifact):</b> Todos los canales presentan una media idéntica a cero ($|\\mu| < 10^{-15}$). Esto confirma formalmente que las señales fueron normalizadas sustrayendo la media temporal de cada ensayo en la etapa previa de procesamiento (<code>d02_processed_data</code>).
    <br/>
    • <b>Dispersión Cinemática:</b> Los sensores de flexión de los dedos (Canales 6 a 10) exhiben una desviación estándar compacta ($\\\\sigma \\approx 10.9 - 12.6$), mientras que el ángulo de Pitch de la muñeca (Canal 11) muestra la mayor amplitud angular ($\\\\sigma = 63.42^\\circ$), capturando los grandes arcos de flexión-extensión en tareas de manipulación y alcance.
    <br/>
    • <b>Asimetría y Curtosis:</b> Los canales de flexión digital presentan asimetría negativa ($-0.88$ a $-1.29$), indicando que la mano permanece en posición neutra la mayor parte del tiempo con excursiones episódicas de flexión. Los canales 4 (IMU2 Yaw, curtosis=82.1) y 5 (IMU2 Roll, curtosis=90.0) muestran colas extremadamente pesadas, originadas por singularidades angulares de Euler o compensaciones espásticas súbitas en el antebrazo.
    """
    story.append(Paragraph(p_univariate, body_style))
    story.append(PageBreak())
    
    # =========================================================
    # PÁGINA 4: FIGURA 2 & ANÁLISIS DE SERIES DE TIEMPO
    # =========================================================
    story.append(Paragraph("3.2 Análisis de Series de Tiempo y Perfiles Cinemáticos Promedio", h2_style))
    p_temporal_intro = """
    A diferencia de datos tabulares tradicionales, los datos de REHAB son <b>series temporales multivariadas continuas</b>. La <b>Figura 2</b> sintetiza las densidades y boxplots de los 12 canales, mientras que la <b>Figura 3</b> despliega las trayectorias dinámicas medias ($\\\\pm 1$ Desviación Estándar) para cuatro movimientos arquetípicos a lo largo de los 880 timesteps.
    """
    story.append(Paragraph(p_temporal_intro, body_style))
    story.append(Spacer(1, 2))
    
    if os.path.exists(figures["fig2"]):
        story.append(Image(figures["fig2"], width=6.8 * inch, height=3.6 * inch))
        story.append(Paragraph("Figura 2: Análisis univariado de densidades empíricas, asimetría, curtosis y boxplots para los 12 canales cinemáticos en el conjunto de entrenamiento (31.4M puntos evaluados).", caption_style))
        story.append(Spacer(1, 4))
        
    p_temporal_insights = """
    <b>Hallazgos Clave en Perfiles Temporales:</b>
    <br/>
    • <b>Mov 00 (Bobath Handshake):</b> Caracterizado por activación suave y coordinada de sensores proximales (IMU1 Pitch), con mínima oscilación en los dedos del guante.
    <br/>
    • <b>Mov 05 (Wrist Flexion & Extension):</b> Muestra oscilaciones periódicas intensas en la muñeca (IMU2) sin reclutamiento de flexión digital aislada.
    <br/>
    • <b>Mov 07 (Ball Gripping):</b> Dominado por flexiones profundas y sostenidas en el Pulgar e Índice, con variaciones de alta amplitud.
    <br/>
    • <b>Mov 15 (Hip Flexion & Extension):</b> Patrón de miembro inferior con grandes desplazamientos angulares en el muslo y reposo en sensores distales.
    """
    story.append(Paragraph(p_temporal_insights, body_style))
    story.append(PageBreak())
    
    # =========================================================
    # PÁGINA 5: FIGURA 3 & ANÁLISIS MULTIVARIADO DE CORRELACIONES
    # =========================================================
    story.append(Paragraph("3.3 Análisis Multivariado: Interacciones Cruzadas y Matrices de Correlación", h2_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=c_secondary, spaceBefore=1, spaceAfter=4))
    
    if os.path.exists(figures["fig3"]):
        story.append(Image(figures["fig3"], width=6.8 * inch, height=3.4 * inch))
        story.append(Paragraph("Figura 3: Perfiles cinemáticos promedio y bandas de variabilidad inter-sujeto (Media ± 1 DE) a lo largo de los 880 timesteps para cuatro movimientos arquetípicos.", caption_style))
        story.append(Spacer(1, 4))
        
    p_multivariate_desc = """
    El análisis multivariado examina la colinealidad, redundancia y acoplamientos biomecánicos entre canales:
    <br/>
    • <b>Sinergia y Acoplamiento Digital (Guante Flex):</b> Se identificó una colinealidad extremadamente alta entre dedos contiguos, destacando <b>Dedo Medio (C08) vs Dedo Anular (C09) con $r = 0.88$</b> ($\\\\rho = 0.86$), y <b>Anular (C09) vs Meñique (C10) con $r = 0.81$</b>. Esto refleja la anatomía funcional de los flexores comunes de los dedos de la mano humana.
    <br/>
    • <b>Coordinación Proximal-Distal (IMUs):</b> Correlación positiva moderada entre Pitch del brazo y antebrazo ($r = 0.64$), capturando arcos de flexión del codo.
    <br/>
    • <b>Desacoplamiento Modal (IMUs vs Guante):</b> Las correlaciones cruzadas entre flexión de dedos e inclinación inercial proximal son cercanas a cero ($|r| < 0.15$), confirmando que <b>ambas modalidades aportan información ortogonal y complementaria</b>.
    """
    story.append(Paragraph(p_multivariate_desc, body_style))
    story.append(PageBreak())
    
    # =========================================================
    # PÁGINA 6: FIGURA 4, FIGURA 5 & DINÁMICA ESPECTRAL FFT
    # =========================================================
    story.append(Paragraph("3.4 Interacciones Cinemáticas 2D y Dinámica Espectral del Movimiento", h2_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=c_secondary, spaceBefore=1, spaceAfter=4))
    
    if os.path.exists(figures["fig4"]):
        story.append(Image(figures["fig4"], width=6.8 * inch, height=2.5 * inch))
        story.append(Paragraph("Figura 4: Matrices de correlación de Pearson (izquierda) y Spearman (derecha) entre los 12 canales sensoriales en el set de entrenamiento.", caption_style))
        story.append(Spacer(1, 2))
        
    if os.path.exists(figures["fig5"]):
        story.append(Image(figures["fig5"], width=6.8 * inch, height=1.9 * inch))
        story.append(Paragraph("Figura 5: Diagramas de dispersión 2D mostrando colinealidad digital (izq), coordinación IMU (centro) y desacoplamiento proximal/distal (der).", caption_style))
        story.append(Spacer(1, 4))
        
    p_spectral_text = """
    <b>Dinámica en Dominio de Frecuencia (FFT / Welch PSD):</b><br/>
    El análisis espectral demuestra que el <b>95% de la energía cinemática voluntaria se concentra por debajo de 3.5 Hz</b>. Las frecuencias superiores a 5.0 Hz corresponden a temblores involuntarios y ruido de cuantización, lo que permite la aplicación segura de filtros pasabajas Butterworth (corte ~ 6 Hz).
    """
    story.append(Paragraph(p_spectral_text, body_style))
    story.append(PageBreak())
    
    # =========================================================
    # PÁGINA 7: FIGURA 6, FIGURA 8 & SEPARABILIDAD LATENTE
    # =========================================================
    story.append(Paragraph("3.5 Análisis Espectral y Separabilidad de Clases en Espacio Latente", h2_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=c_secondary, spaceBefore=1, spaceAfter=4))
    
    if os.path.exists(figures["fig6"]):
        story.append(Image(figures["fig6"], width=6.8 * inch, height=2.4 * inch))
        story.append(Paragraph("Figura 6: Densidad Espectral de Potencia (PSD / Welch) mostrando concentración de energía cinemática en frecuencias bajas (<3.5 Hz).", caption_style))
        story.append(Spacer(1, 2))
        
    if os.path.exists(figures["fig8"]):
        story.append(Image(figures["fig8"], width=6.8 * inch, height=2.6 * inch))
        story.append(Paragraph("Figura 8: Proyecciones en espacio latente (PCA 2D y t-SNE) a partir de 60 features agregadas por ensayo, demostrando clusters compactos y separables.", caption_style))
        story.append(Spacer(1, 4))
        
    p_latent_text = """
    <b>Separabilidad en Espacio Latente:</b> Al proyectar 60 features agregadas por trial mediante t-SNE, las 15 clases forman conglomerados compactos y diferenciables. Se observa solapamiento parcial únicamente dentro de la familia Bobath (Mov 01, 02 y 03) debido a arcos de movimiento compartidos.
    """
    story.append(Paragraph(p_latent_text, body_style))
    story.append(PageBreak())
    
    # =========================================================
    # PÁGINA 8: STEP 3 - DATA QUALITY EVALUATION & FIGURA 7
    # =========================================================
    story.append(Paragraph("4. Paso 3: Evaluación Rigurosa de Calidad de Datos (Data Quality Evaluation)", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=c_secondary, spaceBefore=1, spaceAfter=4))
    
    story.append(Paragraph("4.1 Autopsia Forense del Dato Corrupto: Movimiento 14 (knee flexion and extension)", h2_style))
    p_forensic = """
    <b>Diagnóstico Forense a Nivel de Bytes:</b><br/>
    El archivo oficial <code>014_1.npy</code> (canales inerciales IMU del Movimiento 14) falla al cargarse mediante <code>np.load()</code>. Nuestra inspección revela que los magic bytes estándar <code>b'\\x93NUMPY'</code> fueron reemplazados por <code>b'\\n\\xef\\xbf\\xbdNUMPY'</code>. La secuencia <code>0xEF 0xBF 0xBD</code> corresponde al carácter de sustitución Unicode (<code>U+FFFD</code>), generado cuando un script en el repositorio público de <i>Science Data Bank</i> decodificó incorrectamente el archivo binario como texto UTF-8. Esto causó una pérdida destructiva e irrecuperable de los bytes flotantes, inflando el tamaño a 26.97 MB (frente a 15.16 MB esperados).<br/>
    <b>Veredicto del Senior Analyst:</b> El Movimiento 14 debe ser formalmente excluido del benchmark multimodal de 12 canales.
    """
    forensic_box = Table([[Paragraph(p_forensic, alert_style)]], colWidths=[522])
    forensic_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FFF5F5")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#FEB2B2")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(forensic_box)
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("4.2 Detección de 'Data Smells', Ceros Exactos y Outliers Fisiológicos", h2_style))
    p_smells = """
    • <b>Sesgo de Media Cero (Mean-Centering):</b> La posición angular absoluta fue eliminada en el preprocesamiento; los modelos deben clasificar basándose exclusivamente en la <b>cinemática diferencial relativa</b> (amplitud, RMS y frecuencias).<br/>
    • <b>Ceros Exactos (1.69% de los datos):</b> 531,770 valores numéricos corresponden exactamente a 0.0, reflejando fases de reposo estático antes y después de cada repetición.<br/>
    • <b>Outliers Fisiológicos (3.9% a 19.1%):</b> Corresponden a espasticidad motora y picos de aceleración genuinos de pacientes post-ACV, por lo que no deben ser podados indiscriminadamente.
    """
    story.append(Paragraph(p_smells, body_style))
    story.append(Spacer(1, 4))
    
    if os.path.exists(figures["fig7"]):
        story.append(Image(figures["fig7"], width=6.8 * inch, height=2.3 * inch))
        story.append(Paragraph("Figura 7: Diagnóstico de calidad de datos. Izquierda: Tasas de outliers por IQR y ceros. Derecha: Auditoría de integridad de canales disponibles en los 16 movimientos.", caption_style))
        
    story.append(PageBreak())
    
    # =========================================================
    # PÁGINA 9: RECOMENDACIONES TÉCNICAS, CONCLUSIONES & CIERRE
    # =========================================================
    story.append(Paragraph("5. Recomendaciones Estratégicas y Roadmap Data-Centric para Modelado", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=c_secondary, spaceBefore=1, spaceAfter=4))
    
    rec_data = [
        [Paragraph("Área del Pipeline", table_header), Paragraph("Hallazgo EDA", table_header), Paragraph("Recomendación Técnica del Senior Analyst", table_header)],
        [
            Paragraph("<b>Feature Engineering</b>", table_text),
            Paragraph("Media trial-wise igual a 0. Información en varianza y RMS.", table_text),
            Paragraph("Para modelos tabulares (LightGBM, Random Forest, SVM): Extraer por canal: RMS, Desviación Estándar, Valor Pico a Pico, Energía Espectral (Welch) y Correlaciones Cruzadas (~60 a 96 features consolidadas).", table_text)
        ],
        [
            Paragraph("<b>Deep Learning (1D-CNN / Bi-LSTM)</b>", table_text),
            Paragraph("Señales multivariadas continuas de 880 puntos temporales.", table_text),
            Paragraph("Implementar arquitecturas <b>ResNet-1D</b> o <b>Bi-LSTM</b> sobre el tensor crudo $(N, 880, 12)$ normalizado por canal con Z-Score global del set de entrenamiento.", table_text)
        ],
        [
            Paragraph("<b>Manejo del Movimiento 14</b>", table_text),
            Paragraph("Archivo <code>014_1.npy</code> corrupto en la fuente.", table_text),
            Paragraph("<b>Recomendado:</b> Excluir Movimiento 14 del benchmark de 12 canales (15 clases, N=4,257 total). Reportar experimento secundario solo con guante.", table_text)
        ],
        [
            Paragraph("<b>Esquema de Validación</b>", table_text),
            Paragraph("Desbalance moderado (1.82:1).", table_text),
            Paragraph("Aplicar <b>Stratified 5-Fold Cross Validation</b> garantizando separación por sujeto para evitar fuga de datos clínicos. Evaluar con Macro F1-Score.", table_text)
        ],
        [
            Paragraph("<b>Filtros Digitales</b>", table_text),
            Paragraph("Energía concentrada < 3.5 Hz.", table_text),
            Paragraph("Aplicar filtro pasabajas digital <b>Butterworth 4to orden (fc = 6.0 Hz)</b> para atenuar micro-vibraciones sin alterar la cinemática voluntaria.", table_text)
        ]
    ]
    
    t_rec = Table(rec_data, colWidths=[95, 115, 312])
    t_rec.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_rec)
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("6. Conclusiones y Próximos Pasos del Equipo", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=c_secondary, spaceBefore=1, spaceAfter=4))
    
    p_conclusiones = """
    <b>Síntesis de Hallazgos:</b> El dataset REHAB presenta una excelente viabilidad para clasificación cinemática multiclase. La combinación de IMUs proximales y guante sensorial distal proporciona firmas cinemáticas ortogonales de alta discriminación. La documentación de la media trial-wise en 0 y la degradación de <code>014_1.npy</code> establece bases metodológicas sólidas para el modelado.<br/>
    <b>Próximos Pasos:</b> (1) Construir extractor en <code>src/feature_extractor.py</code>, (2) Entrenar baseline tabular (Random Forest / LightGBM) en <code>notebooks/02_baseline_models.ipynb</code>, y (3) Desarrollar arquitectura 1D-CNN sobre el tensor crudo.
    """
    story.append(Paragraph(p_conclusiones, body_style))
    story.append(Spacer(1, 8))
    
    firma_html = """
    <b>Reporte elaborado y certificado por:</b><br/>
    <b>Lead Data Scientist & Senior Kinematics Analyst (+10 años de experiencia)</b><br/>
    <i>Equipo de Inteligencia Artificial Avanzada para Ciencia de Datos • REHAB-RETO 2026</i>
    """
    story.append(Paragraph(firma_html, callout_style))
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"\nPDF perfeccionado exitosamente en: {output_pdf}")
    return output_pdf


if __name__ == "__main__":
    build_pdf_report()
