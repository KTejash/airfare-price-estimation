"""
Executive 1-Page PDF Generator for Official Airfare Valuation & Quotation Document.
Designed for universal device compatibility (phones, tablets, PCs) with zero missing font glyphs.
Smart India Hackathon (SIH 2026 — SIH26056) | Course: Supervised Learning (INT395)
"""

import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_flight_quotation_pdf(
    origin_city: str,
    dest_city: str,
    src_info: dict,
    dst_info: dict,
    selected_airline: str,
    flight_class: str,
    dep_time: str,
    arr_time: str,
    days_left: int,
    route_info: dict,
    fare_val: float,
    breakdown: dict,
    advisory: dict
) -> bytes:
    """
    Builds a self-contained, universally readable 1-Page Executive PDF Quotation.
    Eliminates Unicode font glyph errors (no missing Rupee glyphs) and fits cleanly on 1 page.
    """
    buffer = io.BytesIO()
    
    # 1-Page tuned page geometry (letter size: 8.5 x 11 inches = 612 x 792 pt)
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=34,
        leftMargin=34,
        topMargin=26,
        bottomMargin=24
    )

    styles = getSampleStyleSheet()

    # Custom Clean Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=17,
        leading=20,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=2
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#2563eb'),
        spaceAfter=6
    )

    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#475569'),
        alignment=2
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=6,
        spaceAfter=4
    )

    cell_bold = ParagraphStyle(
        'CellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1e293b')
    )

    cell_text = ParagraphStyle(
        'CellText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#334155')
    )

    th_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    total_label_style = ParagraphStyle(
        'TotalLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#1d4ed8')
    )

    total_amt_style = ParagraphStyle(
        'TotalAmt',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor('#1d4ed8')
    )

    story = []

    # 1. Header Banner
    gen_time_str = datetime.now().strftime("%d %b %Y, %H:%M:%S")
    ref_num = f"SIH26056-{int(datetime.now().timestamp()) % 1000000:06d}"

    header_left = Paragraph("<b>AIRFARE VALUATION & ITINERARY ESTIMATE</b>", title_style)
    header_sub = Paragraph("Smart India Hackathon (SIH 2026 — SIH26056) • Supervised Learning (INT395)", subtitle_style)
    header_right = Paragraph(f"<b>Ref ID:</b> {ref_num}<br/><b>Date:</b> {gen_time_str}<br/><b>Document:</b> Official Forecast", meta_style)

    header_table = Table([[ [header_left, header_sub], header_right ]], colWidths=[4.4 * inch, 3.1 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563eb'), spaceBefore=3, spaceAfter=8))

    # 2. Section 1: Itinerary & Flight Specifications
    story.append(Paragraph("<b>1. ITINERARY & FLIGHT SPECIFICATIONS</b>", section_heading))
    summary_data = [
        [
            Paragraph("<b>Origin City / Airport:</b>", cell_bold),
            Paragraph(f"{origin_city} ({src_info['code']}) — {src_info['name']}", cell_text),
            Paragraph("<b>Carrier Airline:</b>", cell_bold),
            Paragraph(f"<b>{selected_airline}</b>", cell_text)
        ],
        [
            Paragraph("<b>Destination Airport:</b>", cell_bold),
            Paragraph(f"{dest_city} ({dst_info['code']}) — {dst_info['name']}", cell_text),
            Paragraph("<b>Cabin Class:</b>", cell_bold),
            Paragraph(f"<b>{flight_class} Class</b>", cell_text)
        ],
        [
            Paragraph("<b>Flight Distance:</b>", cell_bold),
            Paragraph(f"{route_info['distance_km']} km ({route_info['distance_miles']} miles)", cell_text),
            Paragraph("<b>Flight Duration:</b>", cell_bold),
            Paragraph(f"{route_info['est_duration']} hrs ({route_info['rec_stops']} stops)", cell_text)
        ],
        [
            Paragraph("<b>Travel Time Windows:</b>", cell_bold),
            Paragraph(f"Departure: {dep_time.replace('_', ' ')} | Arrival: {arr_time.replace('_', ' ')}", cell_text),
            Paragraph("<b>Booking Window:</b>", cell_bold),
            Paragraph(f"<b>{days_left} Days Before Flight</b>", cell_text)
        ]
    ]
    summary_table = Table(summary_data, colWidths=[1.65 * inch, 2.75 * inch, 1.45 * inch, 1.65 * inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(summary_table)

    # 3. Section 2: Itemized Airfare Cost Breakdown (Where Your Money Goes)
    story.append(Paragraph("<b>2. ITEMIZED AIRFARE COST BREAKDOWN (WHERE YOUR MONEY GOES)</b>", section_heading))
    cost_items = [
        [
            Paragraph("<b>Component Description</b>", th_style),
            Paragraph("<b>Aviation Industry Category & Purpose</b>", th_style),
            Paragraph("<b>Share</b>", th_style),
            Paragraph("<b>Amount (INR)</b>", th_style)
        ],
        [
            Paragraph("Base Airfare", cell_bold),
            Paragraph("Aircraft lease, aviation fuel baseline, crew & seat allocation", cell_text),
            Paragraph(f"{(breakdown['Base Fare']/fare_val)*100:.1f}%", cell_text),
            Paragraph(f"INR {breakdown['Base Fare']:,.2f}", cell_bold)
        ],
        [
            Paragraph("Fuel & Demand Surcharge (YQ)", cell_bold),
            Paragraph("Aviation Turbine Fuel (ATF) surcharge & dynamic seat surge", cell_text),
            Paragraph(f"{(breakdown['Fuel & Dynamic Surcharge (YQ)']/fare_val)*100:.1f}%", cell_text),
            Paragraph(f"INR {breakdown['Fuel & Dynamic Surcharge (YQ)']:,.2f}", cell_bold)
        ],
        [
            Paragraph("Airport Development Fee (UDF/PSF)", cell_bold),
            Paragraph("Airport infrastructure, runway safety & passenger terminal facility", cell_text),
            Paragraph(f"{(breakdown['Airport Development Fee (UDF/PSF)']/fare_val)*100:.1f}%", cell_text),
            Paragraph(f"INR {breakdown['Airport Development Fee (UDF/PSF)']:,.2f}", cell_bold)
        ],
        [
            Paragraph("Statutory GST (Tax)", cell_bold),
            Paragraph("Government tax (5% for Economy class / 12% for Business class)", cell_text),
            Paragraph(f"{(breakdown['Goods & Services Tax (GST)']/fare_val)*100:.1f}%", cell_text),
            Paragraph(f"INR {breakdown['Goods & Services Tax (GST)']:,.2f}", cell_bold)
        ],
        [
            Paragraph("<b>TOTAL ESTIMATED FARE</b>", total_label_style),
            Paragraph("<b>All-Inclusive Predicted Ticket Price</b>", cell_bold),
            Paragraph("<b>100.0%</b>", cell_bold),
            Paragraph(f"<b>INR {fare_val:,.2f}</b>", total_amt_style)
        ]
    ]

    bill_table = Table(cost_items, colWidths=[2.3 * inch, 2.7 * inch, 0.8 * inch, 1.7 * inch])
    bill_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#eff6ff')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(bill_table)

    # 4. Section 3: AI Valuation & Booking Advice
    story.append(Paragraph("<b>3. AI VALUATION & TRAVELER BOOKING ADVISORY</b>", section_heading))

    ci_low = max(1500, fare_val - 1819)
    ci_high = fare_val + 1819

    intel_data = [
        [
            Paragraph("<b>AI Model Architecture:</b>", cell_bold),
            Paragraph("Random Forest Regressor (Ensemble ML)", cell_text),
            Paragraph("<b>Model Accuracy (R&sup2;):</b>", cell_bold),
            Paragraph("<b>97.43%</b> (Trained on 300,153 flights)", cell_text)
        ],
        [
            Paragraph("<b>Typical Prediction Error:</b>", cell_bold),
            Paragraph("Average error within &plusmn; INR 1,819.17", cell_text),
            Paragraph("<b>Expected Price Range:</b>", cell_bold),
            Paragraph(f"<b>INR {ci_low:,.0f} – INR {ci_high:,.0f}</b>", cell_text)
        ],
        [
            Paragraph("<b>Booking Recommendation:</b>", cell_bold),
            Paragraph(f"<b>{advisory['status']}</b>", ParagraphStyle('AdvColor', parent=cell_bold, textColor=colors.HexColor('#b91c1c') if days_left <= 7 else colors.HexColor('#15803d'))),
            Paragraph("<b>Actionable Advice:</b>", cell_bold),
            Paragraph(f"{advisory['action']}", cell_text)
        ],
        [
            Paragraph("<b>Carbon Footprint:</b>", cell_bold),
            Paragraph(f"~{route_info['co2_kg']} kg CO<sub>2</sub> emissions per passenger", cell_text),
            Paragraph("<b>Flight Altitude / Speed:</b>", cell_bold),
            Paragraph("FL340 (34,000 ft) at ~Mach 0.78 (830 km/h)", cell_text)
        ]
    ]

    intel_table = Table(intel_data, colWidths=[1.65 * inch, 2.75 * inch, 1.45 * inch, 1.65 * inch])
    intel_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(intel_table)

    # 5. Footer Notice (Compact 1-Page fit)
    footer_text = (
        "<b>Notice:</b> This valuation estimate is generated by a supervised learning regression pipeline trained on "
        "300,153 validated domestic flights for Smart India Hackathon (SIH 2026 — SIH26056). For decision support and price transparency."
    )
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor('#cbd5e1'), spaceBefore=2, spaceAfter=4))
    story.append(Paragraph(footer_text, ParagraphStyle('Foot', parent=styles['Normal'], fontName='Helvetica', fontSize=7.5, leading=10, textColor=colors.HexColor('#64748b'))))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
