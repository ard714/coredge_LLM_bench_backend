"""PDF report generation service for model performance reports."""
from io import BytesIO
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas


class ReportGenerator:
    """Generates PDF performance reports for LLM model evaluations."""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Create custom paragraph styles for the report."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a2e'),
            spaceAfter=12,
            alignment=TA_CENTER,
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#16213e'),
            spaceAfter=10,
            spaceBefore=12,
        ))
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#666666'),
            alignment=TA_CENTER,
            spaceAfter=20,
        ))
        self.styles.add(ParagraphStyle(
            name='MetricLabel',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#666666'),
        ))
        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#1a1a2e'),
            fontName='Helvetica-Bold',
        ))
        self.styles.add(ParagraphStyle(
            name='FooterText',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#999999'),
            alignment=TA_CENTER,
        ))

    def _draw_score_bar(self, score: float, width: float = 100, height: float = 12) -> bytes:
        """Draw a score bar and return as image data."""
        buffer = BytesIO()
        c = canvas.Canvas(buffer, width=width, height=height)

        # Background
        c.setFillColor(colors.HexColor('#e0e0e0'))
        c.roundRect(0, 0, width, height, 6)

        # Fill bar with gradient based on score
        fill_width = width * min(score, 1.0)

        # Color based on score
        if score >= 0.8:
            bar_color = colors.HexColor('#10b981')  # green
        elif score >= 0.6:
            bar_color = colors.HexColor('#3b82f6')  # blue
        elif score >= 0.4:
            bar_color = colors.HexColor('#f59e0b')  # amber
        else:
            bar_color = colors.HexColor('#ef4444')  # red

        c.setFillColor(bar_color)
        c.roundRect(2, 2, fill_width - 4, height - 4, 4)

        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer.getvalue()

    def _format_score_bar_text(self, score: float) -> str:
        """Create a text representation of a score bar."""
        fill_count = int(min(score * 100, 100) / 10)
        # Use ASCII characters that render reliably in PDFs
        filled = "=" * fill_count
        empty = "-" * (10 - fill_count)
        return f"[{filled}{empty}] {score * 100:.0f}%"

    def _create_header(self, title: str, subtitle: str = None) -> list:
        """Create report header elements."""
        elements = []

        # Title
        elements.append(Paragraph(title, self.styles['CustomTitle']))

        if subtitle:
            elements.append(Paragraph(subtitle, self.styles['Subtitle']))

        # Timestamp
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        elements.append(Paragraph(
            f'<para alignment="center" fontSize="9" textColor="#666666">Generated on {timestamp}</para>',
            self.styles['Normal']
        ))
        elements.append(Spacer(1, 0.5 * inch))

        return elements

    def _create_model_summary_table(self, model_data: dict) -> list:
        """Create a summary table with model information."""
        elements = []

        data = [
            ['Model Name', model_data.get('model_name', 'N/A')],
            ['Provider', model_data.get('provider', 'N/A')],
            ['Model ID', model_data.get('model_id', 'N/A')],
            ['Composite Score', f"{model_data.get('composite_score', 0) * 100:.1f}%"],
        ]

        table = Table(data, colWidths=[2 * inch, 3.5 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f5f5')),
            ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#e8f4f8')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1a1a2e')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#1a1a2e')),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_capability_section(self, scores: dict) -> list:
        """Create capability benchmarks section."""
        elements = []
        elements.append(Paragraph("Capability Benchmarks", self.styles['SectionHeader']))

        benchmarks = scores.get('benchmarks', {})

        data = [['Benchmark', 'Score', 'Visual']]
        benchmark_order = ['mmlu', 'gsm8k', 'humaneval']

        for bench_name in benchmark_order:
            if bench_name in benchmarks:
                score = benchmarks[bench_name]
                display_name = bench_name.upper()
                data.append([
                    display_name,
                    f"{score * 100:.1f}%",
                    self._format_score_bar_text(score)
                ])

        # Add average if available
        if 'capability_score' in scores and scores['capability_score'] > 0:
            avg_score = scores['capability_score']
            data.append([
                'Average',
                f"{avg_score * 100:.1f}%",
                self._format_score_bar_text(avg_score)
            ])

        table = Table(data, colWidths=[1.5 * inch, 1 * inch, 2 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('BACKGROUND', (0, -1), (0, -1), colors.HexColor('#f0f0f0')),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_tool_call_section(self, scores: dict) -> list:
        """Create tool call accuracy section."""
        elements = []
        elements.append(Paragraph("Tool Call Accuracy", self.styles['SectionHeader']))

        data = [
            ['Metric', 'Score', 'Visual'],
            ['Accuracy', f"{scores.get('tool_call_score', 0) * 100:.1f}%",
             self._format_score_bar_text(scores.get('tool_call_score', 0))],
        ]

        table = Table(data, colWidths=[1.5 * inch, 1 * inch, 2 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_quality_section(self, scores: dict) -> list:
        """Create quality metrics section."""
        elements = []
        elements.append(Paragraph("Quality Metrics", self.styles['SectionHeader']))

        hallucination = scores.get('hallucination_rate', 0)
        relevancy = scores.get('answer_relevancy', 0)
        faithfulness = scores.get('faithfulness', 0)

        # Quality score is inverse of hallucination combined with relevancy and faithfulness
        quality_score = (relevancy + faithfulness + (1 - hallucination)) / 3 if (relevancy or faithfulness) else 0

        data = [
            ['Metric', 'Score', 'Visual'],
            ['Answer Relevancy', f"{relevancy * 100:.1f}%", self._format_score_bar_text(relevancy)],
            ['Faithfulness', f"{faithfulness * 100:.1f}%", self._format_score_bar_text(faithfulness)],
            ['Hallucination Rate', f"{(1 - hallucination) * 100:.1f}%", self._format_score_bar_text(1 - hallucination)],
            ['Overall Quality', f"{quality_score * 100:.1f}%", self._format_score_bar_text(quality_score)],
        ]

        table = Table(data, colWidths=[1.5 * inch, 1 * inch, 2 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('BACKGROUND', (0, -1), (0, -1), colors.HexColor('#f0f0f0')),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_performance_section(self, scores: dict) -> list:
        """Create performance metrics section."""
        elements = []
        elements.append(Paragraph("Performance Metrics", self.styles['SectionHeader']))

        latency_p50 = scores.get('latency_p50', 0)
        latency_p95 = scores.get('latency_p95', 0)
        tokens_per_sec = scores.get('tokens_per_sec', 0)
        cost_per_1k = scores.get('cost_per_1k', 0)
        error_rate = scores.get('error_rate', 0)

        # Normalize latency for display (lower is better, cap at 5 seconds for scoring)
        latency_score = 1 - min(latency_p50 / 5, 1.0) if latency_p50 else 0

        data = [
            ['Metric', 'Value', 'Notes'],
            ['Latency (P50)', f"{latency_p50:.2f}s" if latency_p50 else 'N/A', 'Lower is better'],
            ['Latency (P95)', f"{latency_p95:.2f}s" if latency_p95 else 'N/A', '95th percentile'],
            ['Latency (P99)', f"{scores.get('latency_p99', 0):.2f}s" if scores.get('latency_p99') else 'N/A', '99th percentile'],
            ['Throughput', f"{tokens_per_sec:.1f} tok/s" if tokens_per_sec else 'N/A', 'Tokens per second'],
            ['Cost', f"${cost_per_1k:.4f}/1K tokens" if cost_per_1k else 'N/A', 'Per 1000 tokens'],
            ['Error Rate', f"{error_rate * 100:.2f}%" if error_rate else '0.00%', 'Request failures'],
        ]

        table = Table(data, colWidths=[1.8 * inch, 1.5 * inch, 1.7 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#f9f9f9')),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _add_footer(self, canvas_obj, doc):
        """Add footer with page number to each page."""
        canvas_obj.saveState()
        canvas_obj.setFont('Helvetica', 8)
        canvas_obj.setFillColor(colors.HexColor('#999999'))
        page_num = doc.page
        canvas_obj.drawCentredString(
            A4[0] / 2,
            0.5 * inch,
            f"Generated by Coredge LLM Benchmark | Page {page_num}"
        )
        canvas_obj.restoreState()

    def generate_single_model_report(self, model_data: dict, scores: dict) -> bytes:
        """
        Generate a PDF report for a single model.

        Args:
            model_data: Dict with model_name, provider, model_id, composite_score
            scores: Dict with all evaluation scores from _get_model_scores()

        Returns:
            BytesIO buffer containing the PDF
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        elements = []

        # Header
        elements.extend(self._create_header(
            "Coredge LLM Benchmark Report",
            "Single Model Performance Analysis"
        ))

        # Model summary
        elements.extend(self._create_model_summary_table(model_data))

        # Capability benchmarks
        if scores.get('capability_score', 0) > 0 or scores.get('benchmarks', {}):
            elements.extend(self._create_capability_section(scores))

        # Tool call accuracy
        if scores.get('tool_call_score', 0) > 0:
            elements.extend(self._create_tool_call_section(scores))

        # Quality metrics
        if scores.get('quality_score', 0) > 0 or scores.get('answer_relevancy', 0) > 0:
            elements.extend(self._create_quality_section(scores))

        # Performance metrics
        if scores.get('tokens_per_sec', 0) > 0 or scores.get('latency_p50', 0) > 0:
            elements.extend(self._create_performance_section(scores))

        # Build PDF
        doc.build(elements, onFirstPage=self._add_footer, onLaterPages=self._add_footer)

        buffer.seek(0)
        return buffer.getvalue()

    def generate_comparison_report(self, models_data: list) -> bytes:
        """
        Generate a PDF comparison report for multiple models.

        Args:
            models_data: List of dicts, each containing model_data and scores

        Returns:
            BytesIO buffer containing the PDF
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=0.5 * inch,
            leftMargin=0.5 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        elements = []

        # Header
        elements.extend(self._create_header(
            "Coredge LLM Benchmark Report",
            f"Model Comparison ({len(models_data)} models)"
        ))

        # Comparison table - overview
        elements.append(Paragraph("Overview Comparison", self.styles['SectionHeader']))

        overview_data = [['Model', 'Composite', 'Capability', 'Tool Call', 'Quality', 'Latency P50', 'Tokens/s']]
        for m in models_data:
            model_data = m.get('model_data', {})
            scores = m.get('scores', {})
            overview_data.append([
                model_data.get('model_name', 'N/A'),
                f"{scores.get('composite_score', 0) * 100:.1f}%",
                f"{scores.get('capability_score', 0) * 100:.1f}%",
                f"{scores.get('tool_call_score', 0) * 100:.1f}%",
                f"{scores.get('quality_score', 0) * 100:.1f}%",
                f"{scores.get('latency_p50', 0):.2f}s" if scores.get('latency_p50') else 'N/A',
                f"{scores.get('tokens_per_sec', 0):.1f}" if scores.get('tokens_per_sec') else 'N/A',
            ])

        col_widths = [1.2 * inch] + [0.9 * inch] * 6
        overview_table = Table(overview_data, colWidths=col_widths)
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fafafa')),
        ]))

        elements.append(overview_table)
        elements.append(Spacer(1, 0.3 * inch))

        # Detailed sections for each model
        for i, m in enumerate(models_data):
            model_data = m.get('model_data', {})
            scores = m.get('scores', {})

            elements.append(Paragraph(
                f"Details: {model_data.get('model_name', 'Model ' + str(i+1))}",
                self.styles['SectionHeader']
            ))

            # Mini summary table
            summary_data = [
                ['Provider', model_data.get('provider', 'N/A')],
                ['Model ID', model_data.get('model_id', 'N/A')],
                ['Composite Score', f"{scores.get('composite_score', 0) * 100:.1f}%"],
            ]
            summary_table = Table(summary_data, colWidths=[1.2 * inch, 2 * inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f5f5')),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ]))
            elements.append(summary_table)
            elements.append(Spacer(1, 0.2 * inch))

            # Capability
            if scores.get('capability_score', 0) > 0 or scores.get('benchmarks', {}):
                elements.append(Paragraph("  Capability Benchmarks", self.styles['MetricLabel']))
                bench_data = [['Benchmark', 'Score']]
                for bench_name in ['mmlu', 'gsm8k', 'humaneval']:
                    if bench_name in scores.get('benchmarks', {}):
                        bench_data.append([
                            bench_name.upper(),
                            f"{scores['benchmarks'][bench_name] * 100:.1f}%"
                        ])
                bench_table = Table(bench_data, colWidths=[1.5 * inch, 1 * inch])
                bench_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8f4f8')),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
                ]))
                elements.append(bench_table)
                elements.append(Spacer(1, 0.15 * inch))

            # Tool call
            if scores.get('tool_call_score', 0) > 0:
                elements.append(Paragraph("  Tool Call Accuracy", self.styles['MetricLabel']))
                tc_data = [['Metric', 'Score']]
                tc_data.append(['Accuracy', f"{scores.get('tool_call_score', 0) * 100:.1f}%"])
                tc_table = Table(tc_data, colWidths=[1.5 * inch, 1 * inch])
                tc_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8f4f8')),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
                ]))
                elements.append(tc_table)
                elements.append(Spacer(1, 0.15 * inch))

            # Quality
            if scores.get('quality_score', 0) > 0 or scores.get('answer_relevancy', 0) > 0:
                elements.append(Paragraph("  Quality Metrics", self.styles['MetricLabel']))
                q_data = [
                    ['Metric', 'Score'],
                    ['Relevancy', f"{scores.get('answer_relevancy', 0) * 100:.1f}%"],
                    ['Faithfulness', f"{scores.get('faithfulness', 0) * 100:.1f}%"],
                    ['Hallucination', f"{scores.get('hallucination_rate', 0) * 100:.1f}%"],
                ]
                q_table = Table(q_data, colWidths=[1.5 * inch, 1 * inch])
                q_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8f4f8')),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
                ]))
                elements.append(q_table)
                elements.append(Spacer(1, 0.15 * inch))

            # Performance
            if scores.get('tokens_per_sec', 0) > 0 or scores.get('latency_p50', 0) > 0:
                elements.append(Paragraph("  Performance", self.styles['MetricLabel']))
                p_data = [
                    ['Metric', 'Value'],
                    ['Latency P50', f"{scores.get('latency_p50', 0):.2f}s"],
                    ['Latency P95', f"{scores.get('latency_p95', 0):.2f}s"],
                    ['Throughput', f"{scores.get('tokens_per_sec', 0):.1f} tok/s"],
                    ['Cost', f"${scores.get('cost_per_1k', 0):.4f}/1K"],
                ]
                p_table = Table(p_data, colWidths=[1.5 * inch, 1 * inch])
                p_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8f4f8')),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
                ]))
                elements.append(p_table)
                elements.append(Spacer(1, 0.3 * inch))

        # Build PDF
        doc.build(elements, onFirstPage=self._add_footer, onLaterPages=self._add_footer)

        buffer.seek(0)
        return buffer.getvalue()


# Singleton instance
_report_generator = None


def get_report_generator() -> ReportGenerator:
    """Get or create the report generator singleton."""
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator
