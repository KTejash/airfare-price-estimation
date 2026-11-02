"""
PDF Generator for Official Airfare Valuation & Quotation Document.
Generates an executive-grade, device-independent PDF report using ReportLab.
SIH 2026 — SIH26056 | Course: Supervised Learning (INT395)
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
    Builds a beautifully styled, high-fidelity PDF quotation byte stream.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'),
        alignment=0,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#2563eb'),
        spaceAfter=12
    )

    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#64748b'),
        alignment=2
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#334155')
    )

    story = []

    # 1. Header Banner Table
    gen_time_str = datetime.now().strftime("%d %b %Y, %H:%M:%S")
    ref_num = f"SIH26056-{int(datetime.now().timestamp()) % 1000000:06d}"

    header_left = Paragraph(
        "<b>AIRFARE VALUATION & ITINERARY QUOTATION</b>",
        title_style
    )
    header_sub = Paragraph(
        "Smart India Hackathon (SIH 2026 — SIH26056) • Supervised Learning (INT395)",
        subtitle_style
    )
    header_right = Paragraph(
        f"<b>Ref ID:</b> {ref_num}<br/><b>Date:</b> {gen_time_str}<br/><b>Status:</b> Official Forecast",
        meta_style
    )

    header_data = [
        [[header_left, header_sub], header_right]
    ]
    header_table = Table(header_data, colWidths=[4.2 * inch, 3.0 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563eb'), spaceBefore=4, spaceAfter=14))

    # 2. Executive Flight Summary Box
    summary_data = [
        [
            Paragraph("<b>Origin City / Airport:</b>", body_style),
            Paragraph(f"{origin_city} ({src_info['code']}) — {src_info['name']}", body_style),
            Paragraph("<b>Carrier Airline:</b>", body_style),
            Paragraph(f"<b>{selected_airline}</b>", body_style)
        ],
        [
            Paragraph("<b>Destination City / Airport:</b>", body_style),
            Paragraph(f"{dest_city} ({dst_info['code']}) — {dst_info['name']}", body_style),
            Paragraph("<b>Cabin Class:</b>", body_style),
            Paragraph(f"<b>{flight_class}</b>", body_style)
        ],
        [
            Paragraph("<b>Great-Circle Distance:</b>", body_style),
            Paragraph(f"{route_info['distance_km']} km ({route_info['distance_miles']} miles)", body_style),
            Paragraph("<b>Flight Duration:</b>", body_style),
            Paragraph(f"{route_info['est_duration']} hrs ({route_info['rec_stops']} stops)", body_style)
        ],
        [
            Paragraph("<b>Time Slots:</b>", body_style),
            Paragraph(f"Dep: {dep_time.replace('_', ' ')} | Arr: {arr_time.replace('_', ' ')}", body_style),
            Paragraph("<b>Booking Window:</b>", body_style),
            Paragraph(f"<b>{days_left} Days Prior to Flight</b>", body_style)
        ]
    ]
    summary_table = Table(summary_data, colWidths=[1.8 * inch, 2.5 * inch, 1.4 * inch, 1.5 * inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(Paragraph("<b>1. ITINERARY & FLIGHT SPECIFICATIONS</b>", section_heading))
    story.append(summary_table)
    story.append(Spacer(1, 12))

    # 3. Itemized Bill of Airfare Table
    story.append(Paragraph("<b>2. ITEMIZED AIRFARE COST BREAKDOWN (BILL OF AIRFARE)</b>", section_heading))

    cost_items = [
        [
            Paragraph("<b>Component Description</b>", body_style),
            Paragraph("<b>Regulatory / Aviation Category</b>", body_style),
            Paragraph("<b>Share (%)</b>", body_style),
            Paragraph("<b>Amount (INR ₹)</b>", body_style)
        ],
        [
            Paragraph("Base Airfare", body_style),
            Paragraph("Fleet Operation, Crew, Aircraft Lease & Margin", body_style),
            Paragraph(f"{(breakdown['Base Fare']/fare_val)*100:.1f}%", body_style),
            Paragraph(f"₹ {breakdown['Base Fare']:,.2f}", body_style)
        ],
        [
            Paragraph("Fuel & Dynamic Demand Surcharge (YQ)", body_style),
            Paragraph("Aviation Turbine Fuel (ATF) & Seat Urgency Factor", body_style),
            Paragraph(f"{(breakdown['Fuel & Dynamic Surcharge (YQ)']/fare_val)*100:.1f}%", body_style),
            Paragraph(f"₹ {breakdown['Fuel & Dynamic Surcharge (YQ)']:,.2f}", body_style)
        ],
        [
            Paragraph("Airport Development Fee (UDF/PSF/ASF)", body_style),
            Paragraph("Airport Infrastructure, Security & Terminal Facility", body_style),
            Paragraph(f"{(breakdown['Airport Development Fee (UDF/PSF)']/fare_val)*100:.1f}%", body_style),
            Paragraph(f"₹ {breakdown['Airport Development Fee (UDF/PSF)']:,.2f}", body_style)
        ],
        [
            Paragraph("Statutory Goods & Services Tax (GST)", body_style),
            Paragraph("Ministry of Finance (5% Economy / 12% Business)", body_style),
            Paragraph(f"{(breakdown['Goods & Services Tax (GST)']/fare_val)*100:.1f}%", body_style),
            Paragraph(f"₹ {breakdown['Goods & Services Tax (GST)']:,.2f}", body_style)
        ],
        [
            Paragraph("<b>TOTAL ESTIMATED AIRFARE</b>", ParagraphStyle('TotalBold', parent=body_style, fontName='Helvetica-Bold', fontSize=10.5, textColor=colors.HexColor('#1d4ed8'))),
            Paragraph("<b>All-Inclusive Commercial Ticket Rate</b>", body_style),
            Paragraph("<b>100.0%</b>", body_style),
            Paragraph(f"<b>₹ {fare_val:,.2f}</b>", ParagraphStyle('TotalBoldAmt', parent=body_style, fontName='Helvetica-Bold', fontSize=11, textColor=colors.HexColor('#1d4ed8')))
        ]
    ]

    bill_table = Table(cost_items, colWidths=[2.5 * inch, 2.5 * inch, 0.9 * inch, 1.3 * inch])
    bill_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#eff6ff')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(bill_table)
    story.append(Spacer(1, 12))

    # 4. Supervised Learning Model & Advisory Table
    story.append(Paragraph("<b>3. ML VALIDATION & YIELD MANAGEMENT INTELLIGENCE</b>", section_heading))

    ci_low = max(1500, fare_val - 1819)
    ci_high = fare_val + 1819

    intel_data = [
        [
            Paragraph("<b>ML Model Architecture:</b>", body_style),
            Paragraph("Random Forest Regressor (Ensemble Bagging)", body_style),
            Paragraph("<b>Model Fidelity ($R^2$):</b>", body_style),
            Paragraph("<b>0.9743</b> (97.43% variance explained)", body_style)
        ],
        [
            Paragraph("<b>Mean Absolute Error:</b>", body_style),
            Paragraph("± ₹ 1,819.17", body_style),
            Paragraph("<b>68% Confidence Band:</b>", body_style),
            Paragraph(f"₹ {ci_low:,.0f} – ₹ {ci_high:,.0f}", body_style)
        ],
        [
            Paragraph("<b>Booking Advisory:</b>", body_style),
            Paragraph(f"<b>{advisory['status']}</b>", ParagraphStyle('Adv', parent=body_style, textColor=colors.HexColor('#b91c1c') if days_left <= 7 else colors.HexColor('#15803d'))),
            Paragraph("<b>Recommendation:</b>", body_style),
            Paragraph(f"{advisory['action']}", body_style)
        ],
        [
            Paragraph("<b>Carbon Footprint ($CO_2$):</b>", body_style),
            Paragraph(f"<b>~{route_info['co2_kg']} kg $CO_2$</b> per passenger", body_style),
            Paragraph("<b>Cruising Altitude:</b>", body_style),
            Paragraph("FL340 (34,000 ft) at ~Mach 0.78", body_style)
        ]
    ]

    intel_table = Table(intel_data, colWidths=[1.8 * inch, 2.5 * inch, 1.4 * inch, 1.5 * inch])
    intel_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(intel_table)
    story.append(Spacer(1, 14))

    # 5. Footer Notice
    footer_text = (
        "<b>Important Notice:</b> This quotation is generated using a supervised learning predictive regression "
        "pipeline trained on 300,153 validated Indian flight records under Smart India Hackathon (SIH 2026 — SIH26056). "
        "Intended for academic benchmarking and consumer price transparency research."
    )
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor('#cbd5e1'), spaceBefore=8, spaceAfter=8))
    story.append(Paragraph(footer_text, ParagraphStyle('FooterStyle', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=8, leading=11, textColor=colors.HexColor('#64748b'))))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
