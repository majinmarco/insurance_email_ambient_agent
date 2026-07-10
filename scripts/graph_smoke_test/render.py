"""Render scenario documents to authentic-looking insurance PDFs with reportlab.

One template per doc type, each returning a list of flowables so several can be
concatenated (separated by ``PageBreak``) into a single PDF — the "multiple doc
types in one attachment" case. Templates mimic real ACORD / carrier documents
(dense boxed grids, letterheads, legal boilerplate, forms schedules, remittance
stubs) via nested reportlab tables.

DOCUMENT TITLE / SEGMENTATION (tier-controlled — see ``realism.py``):
- Each document begins with ONE title line via ``_title_flowable(title, mode)``.
  ``mode="hash"`` emits the literal ``# {TITLE}`` boundary the graph splits bundled
  PDFs on (``graph.py`` splits only single-``#`` H1 lines) — this is the ``clean``
  tier and preserves today's byte-identical rendering.
- The realistic modes (``bold_caps``/``plain``) emit an ordinary title with NO
  leading ``#`` — what real carrier PDFs look like; pdfminer/MarkItDown never turn
  it back into a markdown header. A *single-doc* PDF is one chunk either way, so it
  still segments correctly; a *multi-doc* bundle loses its boundaries and collapses
  to one segment. That collapse is the intended honest signal (``messy`` tier), and
  ``verify.py`` reports it as a ``graph_limitation`` rather than a failure.
- No OTHER rendered line may start with ``# `` (hash+space), in any mode. Section
  headers are bold bars, never H1; render "Invoice No." not "Invoice #". ``_txt``
  strips any leading ``#`` from every dynamic/LLM string, so ``hash`` mode keeps
  exactly one boundary per document.
- Faux logos are vector table cells, never raster Images (avoids MarkItDown's LLM
  image captioner injecting a stray line).
"""

from __future__ import annotations

import io
import re
from collections import Counter

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from insurance_email_agent.schemas import DocumentCategory

from .scenario import (
    CoverageLine,
    DocumentContent,
    LimitLine,
    SharedFacts,
    TableRow,
)

DC = DocumentCategory

# --------------------------------------------------------------------------- #
# Palette & styles
# --------------------------------------------------------------------------- #
_RULE = colors.black
_LABELBG = colors.HexColor("#e6e6e6")
_NAVY = colors.HexColor("#0b3d6b")
_SHADE = colors.HexColor("#f2f5f8")
_HEADBG = colors.HexColor("#dce6f1")
_USABLE_W = letter[0] - 1.0 * inch  # 0.5in margins → 7.5in

_S = getSampleStyleSheet()
_H1 = ParagraphStyle("H1", parent=_S["Heading1"], fontName="Helvetica-Bold", fontSize=13, spaceBefore=0, spaceAfter=6)
_LABEL = ParagraphStyle("Label", parent=_S["Normal"], fontName="Helvetica", fontSize=5.5, textColor=colors.HexColor("#333333"), leading=6.5)
_CELL = ParagraphStyle("Cell", parent=_S["Normal"], fontName="Helvetica", fontSize=7.5, leading=9)
_CELL_B = ParagraphStyle("CellB", parent=_CELL, fontName="Helvetica-Bold")
_CELL_R = ParagraphStyle("CellR", parent=_CELL, alignment=TA_RIGHT)
_TINY = ParagraphStyle("Tiny", parent=_S["Normal"], fontName="Helvetica", fontSize=6, leading=7)
_BODY = ParagraphStyle("Body", parent=_S["Normal"], fontName="Helvetica", fontSize=8.5, leading=11.5, alignment=TA_JUSTIFY, spaceAfter=6)
_DISCLAIMER = ParagraphStyle("Disc", parent=_S["Normal"], fontName="Helvetica", fontSize=6.3, leading=7.6, alignment=TA_JUSTIFY, spaceAfter=4)
_SECTION = ParagraphStyle("Section", parent=_S["Normal"], fontName="Helvetica-Bold", fontSize=8.5, textColor=colors.black, leading=11)
_SECTION_W = ParagraphStyle("SectionW", parent=_SECTION, textColor=colors.white)
_BIG = ParagraphStyle("Big", parent=_S["Normal"], fontName="Helvetica-Bold", fontSize=12, leading=14)
_SUB = ParagraphStyle("Sub", parent=_S["Normal"], fontName="Helvetica", fontSize=8, textColor=colors.grey)
_MONO = ParagraphStyle("Mono", parent=_S["Normal"], fontName="Helvetica-Bold", fontSize=17, textColor=colors.white, alignment=TA_CENTER, leading=19)
_LOGONAME = ParagraphStyle("LogoName", parent=_S["Normal"], fontName="Helvetica-Bold", fontSize=13, leading=15)

_COPYRIGHT = "Synthetic document generated for pipeline testing — not a real policy."


# --------------------------------------------------------------------------- #
# Text sanitizing (protects the single-# segmentation invariant)
# --------------------------------------------------------------------------- #
def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _sanitize(s: str) -> str:
    """Drop any leading run of '#' (+spaces) and XML-escape."""
    return _esc(re.sub(r"^\s*#+\s*", "", s))


def _txt(s: str | None, style: ParagraphStyle = _CELL, placeholder: str = "—") -> Paragraph:
    raw = s if (s is not None and str(s).strip() != "") else placeholder
    return Paragraph(_sanitize(str(raw)).replace("\n", "<br/>"), style)


def _title_flowable(title: str, mode: str = "hash") -> Paragraph:
    """The per-document title line — the ONLY place a '# ' boundary may appear.

    ``mode``: ``"hash"`` → the splittable ``# TITLE`` H1 (today's behavior);
    ``"bold_caps"``/``"plain"`` → a realistic title with NO leading ``#`` (real
    carrier PDFs have no markdown header). See the module docstring / ``realism.py``.
    """
    t = _sanitize(title)
    if mode == "hash":
        return Paragraph("# " + t, _H1)
    if mode == "bold_caps":
        return Paragraph(f"<b>{t.upper()}</b>", _H1)
    return Paragraph(t, _H1)  # "plain"


def _mdy(iso: str | None) -> str:
    if not iso:
        return "—"
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", iso.strip())
    return f"{m.group(2)}/{m.group(3)}/{m.group(1)}" if m else iso


def _sum_money(values) -> str:
    """Sum a list of '$1,234.00'-style strings → '$X,XXX.XX' (so totals reconcile)."""
    total = 0.0
    for v in values:
        if not v:
            continue
        try:
            total += float(re.sub(r"[^0-9.\-]", "", v))
        except ValueError:
            pass
    return f"${total:,.2f}"


# --------------------------------------------------------------------------- #
# Generic table / box helpers
# --------------------------------------------------------------------------- #
def _grid(data, col_widths, *, header_rows=0, extra=None, font_size=7.0, split=False) -> Table:
    """Dense hairline ACORD-style grid."""
    t = Table(data, colWidths=col_widths, repeatRows=header_rows if split else 0)
    ops = [
        ("GRID", (0, 0), (-1, -1), 0.4, _RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    if header_rows:
        ops.append(("BACKGROUND", (0, 0), (-1, header_rows - 1), _HEADBG))
    if extra:
        ops.extend(extra)
    t.setStyle(TableStyle(ops))
    return t


def _labeled_box(label: str, inner, *, width: float | None = None) -> Table:
    """Bordered box: shaded caption strip over a content row holding `inner`."""
    t = Table([[Paragraph(_sanitize(label), _LABEL)], [inner]], colWidths=[width or _USABLE_W])
    t.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, _RULE),
                ("LINEBELOW", (0, 0), (0, 0), 0.4, _RULE),
                ("BACKGROUND", (0, 0), (0, 0), _LABELBG),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def _section(title: str) -> Table:
    """Full-width navy bold section bar with white text (never an H1)."""
    t = Table([[Paragraph(_sanitize(title), _SECTION_W)]], colWidths=[_USABLE_W])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def _letterhead(carrier: str, *, tagline: str | None = None, address: str | None = None) -> Table:
    initials = "".join(w[0] for w in re.findall(r"[A-Za-z]+", carrier)[:3]).upper() or "INS"
    right_lines = [f"<b>{_sanitize(carrier)}</b>"]
    if tagline:
        right_lines.append(_sanitize(tagline))
    if address:
        right_lines.append(_sanitize(address).replace("\n", "<br/>"))
    right = Paragraph("<br/>".join(right_lines), _LOGONAME)
    mono = Paragraph(initials, _MONO)
    t = Table([[mono, right]], colWidths=[0.95 * inch, _USABLE_W - 0.95 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), _NAVY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (1, 0), (1, 0), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, -1), 1.2, _NAVY),
            ]
        )
    )
    return t


def _disclaimer(text: str) -> Paragraph:
    return Paragraph(_sanitize(text), _DISCLAIMER)


# --------------------------------------------------------------------------- #
# Domain table builders
# --------------------------------------------------------------------------- #
_STD_TYPE_SUBLINES: dict[str, list[str]] = {
    "general liability": ["[X] Occurrence   [ ] Claims-Made", "Gen'l Aggregate Limit Applies Per:", "[ ] Policy  [ ] Project  [ ] Loc"],
    "automobile": ["[X] Any Auto", "[ ] Owned  [ ] Hired  [ ] Non-Owned Autos"],
    "umbrella": ["[X] Occurrence   [ ] Claims-Made", "[ ] Retention"],
    "excess": ["[X] Occurrence   [ ] Claims-Made", "[ ] Retention"],
    "workers": ["[X] Per Statute   [ ] Other"],
}


def _type_sublines(coverage_type: str | None) -> list[str]:
    key = (coverage_type or "").lower()
    for k, v in _STD_TYPE_SUBLINES.items():
        if k in key:
            return v
    return []


def _mini_limits(lines: list[LimitLine] | None) -> Table:
    data = [[_txt(ln.coverage, _TINY), _txt(ln.limit, ParagraphStyle("TinyR", parent=_TINY, alignment=TA_RIGHT))] for ln in (lines or [])]
    if not data:
        data = [[_txt("", _TINY), _txt("", _TINY)]]
    t = Table(data, colWidths=[1.12 * inch, 0.62 * inch])
    t.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 1),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ]
        )
    )
    return t


def _coverages_grid(covs: list[CoverageLine]) -> Table:
    header = [
        _txt("INSR LTR", _CELL_B), _txt("TYPE OF INSURANCE", _CELL_B),
        _txt("ADDL INSD", _CELL_B), _txt("SUBR WVD", _CELL_B),
        _txt("POLICY NUMBER", _CELL_B), _txt("POLICY EFF", _CELL_B),
        _txt("POLICY EXP", _CELL_B), _txt("LIMITS", _CELL_B),
    ]
    rows = [header]
    for c in covs:
        type_cell = [Paragraph(f"<b>{_sanitize(c.coverage_type or '')}</b>", _TINY)]
        type_cell += [Paragraph(_sanitize(s), _TINY) for s in _type_sublines(c.coverage_type)]
        rows.append(
            [
                _txt(c.insurer_letter, _CELL),
                type_cell,
                _txt("X" if c.addl_insured else "", _CELL),
                _txt("X" if c.subro_waived else "", _CELL),
                _txt(c.policy_number, _TINY),
                _txt(c.effective_date, _TINY),
                _txt(c.expiration_date, _TINY),
                _mini_limits(c.limits),
            ]
        )
    widths = [0.35 * inch, 1.9 * inch, 0.4 * inch, 0.4 * inch, 1.1 * inch, 0.75 * inch, 0.75 * inch, 1.85 * inch]
    return _grid(rows, widths, header_rows=1, font_size=6.5, split=True)


def _coverage_parts_table(parts: list[CoverageLine], *, total: str | None = None) -> Table:
    rows = [[_txt("COVERAGE PART", _CELL_B), _txt("LIMITS", _CELL_B), _txt("PREMIUM", _CELL_B)]]
    for p in parts:
        limits = "<br/>".join(f"{_sanitize(l.coverage)}: {_sanitize(l.limit)}" for l in (p.limits or []))
        rows.append([_txt(p.coverage_type, _CELL_B), Paragraph(limits or "—", _CELL), _txt(p.premium, _CELL_R)])
    if total:
        rows.append([_txt("Total Policy Premium", _CELL_B), _txt(""), _txt(total, _CELL_R)])
    widths = [2.5 * inch, 3.4 * inch, 1.6 * inch]
    extra = [("BACKGROUND", (0, len(rows) - 1), (-1, len(rows) - 1), _SHADE)] if total else None
    return _grid(rows, widths, header_rows=1, font_size=7.5, extra=extra)


def _forms_table(forms) -> Table:
    rows = [[_txt("FORM NUMBER", _CELL_B), _txt("TITLE", _CELL_B)]]
    for f in forms or []:
        rows.append([_txt(f.form_number, _CELL), _txt(f.title, _CELL)])
    return _grid(rows, [1.6 * inch, 5.9 * inch], header_rows=1, font_size=7.5, split=True)


def _charges_table(items, *, total_label: str | None = None, total_amount: str | None = None) -> Table:
    rows = [[_txt("DESCRIPTION", _CELL_B), _txt("AMOUNT", ParagraphStyle("AmtH", parent=_CELL_B, alignment=TA_RIGHT))]]
    for it in items or []:
        rows.append([_txt(it.description, _CELL), _txt(it.amount, _CELL_R)])
    if total_amount:
        rows.append([_txt(total_label or "TOTAL", _CELL_B), _txt(total_amount, ParagraphStyle("AmtT", parent=_CELL_R, fontName="Helvetica-Bold"))])
    widths = [5.7 * inch, 1.8 * inch]
    extra = [("BACKGROUND", (0, len(rows) - 1), (-1, len(rows) - 1), _SHADE)] if total_amount else None
    return _grid(rows, widths, header_rows=1, font_size=7.5, extra=extra)


def _schedule_table(rows) -> Table:
    data = [[_txt(r.label, _CELL_B), _txt(r.value, _CELL)] for r in (rows or [])]
    if not data:
        data = [[_txt("", _CELL), _txt("", _CELL)]]
    return _grid(data, [2.2 * inch, 5.3 * inch], font_size=7.5)


def _amount_due_box(amount: str | None, due_date: str | None) -> Table:
    t = Table(
        [
            [Paragraph("TOTAL AMOUNT DUE", _CELL_B), Paragraph(_sanitize(amount or "—"), ParagraphStyle("DueAmt", parent=_BIG, alignment=TA_RIGHT))],
            [Paragraph("Payment Due Date", _CELL), Paragraph(_sanitize(due_date or "—"), _CELL_R)],
        ],
        colWidths=[5.7 * inch, 1.8 * inch],
    )
    t.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 1.2, _NAVY),
                ("BACKGROUND", (0, 0), (-1, 0), _HEADBG),
                ("LINEBELOW", (0, 0), (-1, 0), 0.4, _RULE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def _perforation(label: str = "DETACH AND RETURN WITH PAYMENT") -> Table:
    t = Table([[Paragraph(_sanitize(label), _TINY)]], colWidths=[_USABLE_W])
    t.setStyle(
        TableStyle(
            [
                ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.grey, 1, (2, 2)),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def _remittance_stub(shared: SharedFacts, doc: DocumentContent) -> Table:
    inner = _grid(
        [
            [_txt("Invoice No.", _CELL_B), _txt(doc.invoice_number, _CELL), _txt("Account No.", _CELL_B), _txt(doc.account_number, _CELL)],
            [_txt("Policy No.", _CELL_B), _txt(shared.policy_number or "PENDING", _CELL), _txt("Due Date", _CELL_B), _txt(doc.due_date, _CELL)],
            [_txt("Amount Due", _CELL_B), _txt(doc.amount_due, _CELL_B), _txt("Insured", _CELL_B), _txt(shared.named_insured, _CELL)],
        ],
        [0.9 * inch, 2.0 * inch, 0.9 * inch, 2.8 * inch],
        font_size=7.5,
    )
    remit = _txt("Remit To: " + (doc.remit_to or shared.carrier_name), _CELL)
    return _labeled_box("REMITTANCE — RETURN THIS PORTION WITH YOUR PAYMENT", Table([[inner], [remit]], colWidths=[_USABLE_W - 0.06 * inch], style=TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 1), (-1, 1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)])))


def _endorsement_banner() -> Table:
    t = Table([[Paragraph("THIS ENDORSEMENT CHANGES THE POLICY. PLEASE READ IT CAREFULLY.", _CELL_B)]], colWidths=[_USABLE_W])
    t.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 1.0, _RULE),
                ("BACKGROUND", (0, 0), (-1, -1), _LABELBG),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


# --------------------------------------------------------------------------- #
# Footer: NumberedCanvas + per-document _FooterCue
# --------------------------------------------------------------------------- #
class _FooterCue(Flowable):
    """Zero-size flowable that stamps the owning doc's footer text on the canvas."""

    def __init__(self, form_number: str, doc_key: int):
        super().__init__()
        self.form_number = form_number
        self.doc_key = doc_key

    def wrap(self, *_):
        return (0, 0)

    def draw(self):
        self.canv._doc_form = self.form_number
        self.canv._doc_key = self.doc_key


class _WatermarkCue(Flowable):
    """Zero-size flowable that sets (or clears) the owning doc's watermark text.

    Emitted once at the start of every document so a watermark never bleeds into the
    next document's pages. Drawn as vector text (see ``_draw_watermark``) so it stays
    in the extracted markdown as realistic classifier noise and never triggers
    MarkItDown's image captioner.
    """

    def __init__(self, text: str | None):
        super().__init__()
        self.text = text

    def wrap(self, *_):
        return (0, 0)

    def draw(self):
        self.canv._doc_watermark = self.text


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas: per-document 'Page i of n' + form-number footers + watermark."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._saved = []

    def showPage(self):
        self._saved.append({
            **self.__dict__,
            "_form": getattr(self, "_doc_form", ""),
            "_key": getattr(self, "_doc_key", 0),
            "_wm": getattr(self, "_doc_watermark", None),
        })
        self._startPage()

    def save(self):
        totals = Counter(s["_key"] for s in self._saved)
        seen: dict[int, int] = {}
        for st in self._saved:
            self.__dict__.update(st)
            k = st["_key"]
            seen[k] = seen.get(k, 0) + 1
            if st.get("_wm"):
                self._draw_watermark(st["_wm"])
            self._draw_footer(st["_form"], seen[k], totals[k])
            super().showPage()
        super().save()

    def _draw_watermark(self, text: str):
        # Horizontal (not rotated): a 45° stamp is extracted glyph-by-glyph by pdfminer
        # (each rotated glyph lands on its own baseline), which loses the word. Drawn
        # flat, the whole token survives into the markdown as realistic classifier noise.
        self.saveState()
        self.setFont("Helvetica-Bold", 60)
        self.setFillColor(colors.HexColor("#9a9a9a"))
        try:
            self.setFillAlpha(0.13)
        except Exception:  # noqa: BLE001 — older reportlab without alpha; light grey still reads
            pass
        self.drawCentredString(letter[0] / 2, letter[1] / 2, text)
        self.restoreState()

    def _draw_footer(self, form: str, i: int, n: int):
        y = 0.32 * inch
        self.setStrokeColor(colors.grey)
        self.setLineWidth(0.4)
        self.line(0.5 * inch, y + 9, letter[0] - 0.5 * inch, y + 9)
        self.setFont("Helvetica", 6.5)
        self.setFillColor(colors.HexColor("#555555"))
        if form:
            self.drawString(0.5 * inch, y, form)
        self.drawCentredString(letter[0] / 2, y, _COPYRIGHT)
        self.drawRightString(letter[0] - 0.5 * inch, y, f"Page {i} of {n}")


# --------------------------------------------------------------------------- #
# Per-doc-type templates  (each: _FooterCue → _title_flowable → body)
# --------------------------------------------------------------------------- #
def _producer_para(shared: SharedFacts) -> Paragraph:
    p = shared.producer
    if not p:
        return _txt(shared.carrier_name, _CELL)
    lines = [f"<b>{_sanitize(p.agency or '')}</b>"]
    if p.address:
        lines.append(_sanitize(p.address))
    contact = []
    if p.contact_name:
        contact.append(f"CONTACT: {_sanitize(p.contact_name)}")
    if p.phone:
        contact.append(f"PHONE: {_sanitize(p.phone)}")
    if p.fax:
        contact.append(f"FAX: {_sanitize(p.fax)}")
    if contact:
        lines.append("  ".join(contact))
    if p.email:
        lines.append(f"E-MAIL: {_sanitize(p.email)}")
    return Paragraph("<br/>".join(lines), _CELL)


def _insured_para(shared: SharedFacts) -> Paragraph:
    lines = [f"<b>{_sanitize(shared.named_insured)}</b>", _sanitize(shared.insured_address).replace("\n", "<br/>")]
    if shared.fein:
        lines.append(f"FEIN: {_sanitize(shared.fein)}")
    return Paragraph("<br/>".join(lines), _CELL)


def _insurers_para(shared: SharedFacts) -> Paragraph:
    lines = []
    for ins in shared.insurers or []:
        lines.append(f"INSURER {_sanitize(ins.letter or '')}: {_sanitize(ins.name or '')}  NAIC No. {_sanitize(ins.naic or '')}")
    if not lines:
        lines = [f"INSURER A: {_sanitize(shared.carrier_name)}"]
    return Paragraph("<br/>".join(lines), _CELL)


def render_certificate(shared: SharedFacts, doc: DocumentContent, doc_index: int = 0, *, title_mode: str = "hash", layout: int = 0) -> list:
    covs = doc.coverage_lines or [
        CoverageLine(insurer_letter="A", coverage_type="Commercial General Liability",
                     policy_number=shared.policy_number, effective_date=_mdy(shared.effective_date),
                     expiration_date=_mdy(shared.expiration_date), limits=list(shared.coverage_limits))
    ]
    date_box = Table([[Paragraph("DATE (MM/DD/YYYY)", _LABEL)], [Paragraph(_mdy(shared.effective_date), _CELL)]], colWidths=[1.5 * inch])
    date_box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, _RULE), ("BACKGROUND", (0, 0), (0, 0), _LABELBG), ("LEFTPADDING", (0, 0), (-1, -1), 3)]))
    top = Table([[Paragraph("Issued as a matter of information only.", _SUB), date_box]], colWidths=[_USABLE_W - 1.6 * inch, 1.6 * inch])
    top.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (1, 0), (1, 0), "RIGHT")]))

    lw = 4.3 * inch
    header_band = Table(
        [
            [_labeled_box("PRODUCER", _producer_para(shared), width=lw), _labeled_box("INSURER(S) AFFORDING COVERAGE", _insurers_para(shared), width=_USABLE_W - lw)],
            [_labeled_box("INSURED", _insured_para(shared), width=lw), ""],
        ],
        colWidths=[lw, _USABLE_W - lw],
    )
    header_band.setStyle(TableStyle([("SPAN", (1, 0), (1, 1)), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))

    holder = _labeled_box("CERTIFICATE HOLDER", _txt(doc.certificate_holder, _CELL), width=3.7 * inch)
    cancel = _labeled_box("CANCELLATION", _disclaimer(_CANCELLATION_TEXT), width=_USABLE_W - 3.8 * inch)
    bottom = Table([[holder, cancel]], colWidths=[3.7 * inch, _USABLE_W - 3.7 * inch])
    bottom.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    sig = _labeled_box(
        "AUTHORIZED REPRESENTATIVE",
        Paragraph(f"{_sanitize(doc.authorized_representative or (shared.producer.contact_name if shared.producer else ''))}<br/>_____________________________", _CELL),
    )

    return [
        _FooterCue("ACORD 25 (2016/03)  ·  © 1988-2015 ACORD CORPORATION. All rights reserved.", doc_index),
        _title_flowable("CERTIFICATE OF LIABILITY INSURANCE", title_mode),
        top,
        Spacer(1, 3),
        _disclaimer(_ACORD_DISCLAIMER),
        _disclaimer(_ACORD_IMPORTANT),
        Spacer(1, 3),
        header_band,
        Spacer(1, 4),
        _section(f"COVERAGES    CERTIFICATE NUMBER: {doc.certificate_number or 'N/A'}    REVISION NUMBER: {doc.revision_number or '0'}"),
        _coverages_grid(covs),
        Spacer(1, 4),
        _labeled_box("DESCRIPTION OF OPERATIONS / LOCATIONS / VEHICLES", _txt(doc.description_of_operations, _CELL)),
        Spacer(1, 4),
        KeepTogether([bottom, Spacer(1, 3), sig]),
    ]


def render_declarations(shared: SharedFacts, doc: DocumentContent, doc_index: int = 0, *, title_mode: str = "hash", layout: int = 0) -> list:
    period = f"{_mdy(doc.effective_date or shared.effective_date)} to {_mdy(doc.expiration_date or shared.expiration_date)}, 12:01 A.M. Standard Time at the mailing address of the Named Insured"
    info = _grid(
        [
            [_txt("POLICY NUMBER", _LABEL), _txt(shared.policy_number or "PENDING", _CELL_B), _txt("POLICY PERIOD", _LABEL), _txt(period, _CELL)],
            [_txt("NAMED INSURED", _LABEL), Paragraph(f"<b>{_sanitize(shared.named_insured)}</b><br/>{_sanitize(shared.insured_address)}", _CELL), _txt("FORM OF BUSINESS", _LABEL), _txt(shared.entity_type, _CELL)],
            [_txt("PRODUCER", _LABEL), _txt(shared.producer.agency if shared.producer else shared.carrier_name, _CELL), _txt("BUSINESS DESCRIPTION", _LABEL), _txt(shared.business_description, _CELL)],
        ],
        [1.1 * inch, 2.65 * inch, 1.25 * inch, 2.5 * inch],
        font_size=7.5,
    )
    parts = doc.coverage_parts or []
    tagline = ("Commercial Lines Division", "Commercial Lines Underwriting", "Business Insurance Division")[layout % 3]
    story = [
        _FooterCue(f"POLICY NO. {shared.policy_number or 'PENDING'}  ·  COMMERCIAL PACKAGE DECLARATIONS", doc_index),
        _title_flowable("COMMERCIAL PACKAGE POLICY DECLARATIONS", title_mode),
        _letterhead(shared.carrier_name, tagline=tagline, address="A Stock Insurance Company · 151 N Franklin St, Chicago, IL 60606"),
        Spacer(1, 5),
        info,
        Spacer(1, 6),
        _section("SCHEDULE OF COVERAGE PARTS"),
        _coverage_parts_table(parts, total=_sum_money([p.premium for p in parts]) if parts else doc.total_premium),
        Spacer(1, 6),
    ]
    premium_block = (
        [_section("PREMIUM SUMMARY"), _charges_table(doc.premium_summary, total_label="Total Amount", total_amount=doc.total_premium), Spacer(1, 6)]
        if doc.premium_summary else []
    )
    forms_block = (
        [_section("SCHEDULE OF FORMS AND ENDORSEMENTS"), _forms_table(doc.forms_schedule), Spacer(1, 6)]
        if doc.forms_schedule else []
    )
    # layout variant changes the section order (a real text-order change downstream).
    story += (forms_block + premium_block) if layout == 1 else (premium_block + forms_block)
    countersig = Paragraph(
        f"Countersigned by: <b>{_sanitize(doc.countersignature or shared.carrier_name)}</b><br/>"
        f"Date: {_sanitize(doc.countersignature_date or shared.effective_date)}<br/>"
        "_____________________________  (Authorized Representative)",
        _CELL,
    )
    story.append(KeepTogether(_labeled_box("COUNTERSIGNATURE", countersig)))
    return story


def render_invoice(shared: SharedFacts, doc: DocumentContent, doc_index: int = 0, *, title_mode: str = "hash", layout: int = 0) -> list:
    info = _grid(
        [
            [_txt("INVOICE NO.", _LABEL), _txt(doc.invoice_number, _CELL_B), _txt("INVOICE DATE", _LABEL), _txt(doc.invoice_date, _CELL)],
            [_txt("ACCOUNT NO.", _LABEL), _txt(doc.account_number, _CELL), _txt("POLICY NO.", _LABEL), _txt(shared.policy_number or "PENDING", _CELL)],
            [_txt("BILLING PERIOD", _LABEL), _txt(doc.billing_period, _CELL), _txt("DUE DATE", _LABEL), _txt(doc.due_date, _CELL)],
        ],
        [1.1 * inch, 2.65 * inch, 1.1 * inch, 2.65 * inch],
        font_size=7.5,
    )
    bill_to = _labeled_box("BILL TO", Paragraph(f"<b>{_sanitize(shared.named_insured)}</b><br/>{_sanitize(shared.insured_address)}", _CELL))
    remit = doc.remit_to or shared.carrier_name
    if remit.startswith(shared.carrier_name):  # avoid repeating the carrier name in the letterhead
        remit = remit[len(shared.carrier_name):].strip("\n ,") or "PO Box 74007619, Chicago, IL 60674"
    return [
        _FooterCue(f"BILLING STATEMENT  ·  INVOICE {doc.invoice_number or ''}", doc_index),
        _title_flowable("PREMIUM INVOICE", title_mode),
        _letterhead(shared.carrier_name, tagline="Billing Department", address="Remittance: " + remit),
        Spacer(1, 5),
        info,
        Spacer(1, 5),
        bill_to,
        Spacer(1, 6),
        _section("ITEMIZED CHARGES"),
        _charges_table(doc.line_items, total_label="Total Charges", total_amount=doc.amount_due),
        Spacer(1, 6),
        _amount_due_box(doc.amount_due, doc.due_date),
        Spacer(1, 6),
        _labeled_box("PAYMENT TERMS & METHODS", _txt(doc.payment_methods, _CELL)),
        Spacer(1, 14),
        _perforation(),
        Spacer(1, 4),
        KeepTogether(_remittance_stub(shared, doc)),
    ]


def render_endorsement(shared: SharedFacts, doc: DocumentContent, doc_index: int = 0, *, title_mode: str = "hash", layout: int = 0) -> list:
    info = _grid(
        [
            [_txt("ENDORSEMENT EFFECTIVE DATE", _LABEL), _txt(doc.effective_date or shared.effective_date, _CELL_B), _txt("ENDORSEMENT NO.", _LABEL), _txt(doc.endorsement_number, _CELL)],
            [_txt("POLICY NUMBER", _LABEL), _txt(shared.policy_number or "PENDING", _CELL), _txt("FORM NUMBER", _LABEL), _txt(doc.form_number, _CELL)],
            [_txt("NAMED INSURED", _LABEL), _txt(shared.named_insured, _CELL), _txt("CARRIER", _LABEL), _txt(shared.carrier_name, _CELL)],
        ],
        [1.5 * inch, 2.25 * inch, 1.25 * inch, 2.5 * inch],
        font_size=7.5,
    )
    story = [
        _FooterCue(f"{doc.form_number or 'CHANGE ENDORSEMENT'}  ·  POLICY NO. {shared.policy_number or 'PENDING'}", doc_index),
        _title_flowable("POLICY CHANGE ENDORSEMENT", title_mode),
        _endorsement_banner(),
        Spacer(1, 5),
        Paragraph(f"<b>{_sanitize(doc.endorsement_title or 'Policy Change')}</b>", _SECTION),
        Paragraph("This endorsement modifies insurance provided under the Business Auto / Commercial Package Policy.", _SUB),
        Spacer(1, 5),
        info,
        Spacer(1, 6),
    ]
    if doc.schedule_rows:
        story += [KeepTogether(_labeled_box("SCHEDULE", _schedule_table(doc.schedule_rows))), Spacer(1, 6)]
    story.append(_section("AMENDATORY PROVISIONS"))
    for i, prov in enumerate(doc.provisions or [doc.change_description or "See schedule for amended terms."], start=1):
        story.append(Paragraph(f"{i}.&nbsp; {_sanitize(prov)}", _BODY))
    story += [
        Spacer(1, 4),
        _labeled_box("PREMIUM", _txt(f"Additional Premium: {doc.additional_premium or '$0.00'}", _CELL)),
        Spacer(1, 4),
        Paragraph("All other terms and conditions of this policy remain unchanged.", _BODY),
    ]
    return story


def _data_table(columns: list[str] | None, rows: list[TableRow] | None) -> Table:
    """Generic column/row data table (loss-run claims, SOV schedule, etc.)."""
    cols = columns or ["Item", "Detail"]
    n = len(cols)
    data = [[_txt(c, _CELL_B) for c in cols]]
    for r in rows or []:
        cells = (list(r.cells) + [""] * n)[:n]
        data.append([_txt(c, _CELL) for c in cells])
    return _grid(data, [_USABLE_W / n] * n, header_rows=1, font_size=7.0, split=True)


def render_needs_review(shared: SharedFacts, doc: DocumentContent, doc_index: int = 0, *, title_mode: str = "hash", layout: int = 0) -> list:
    """A structured, insurance-adjacent document that is NOT one of the four modeled
    doctypes (loss run, application, statement of values, cover letter) — should route
    to needs_review."""
    title = doc.nr_kind or doc.title or "Supplemental Insurance Document"
    story = [
        _FooterCue((doc.nr_kind or "SUPPLEMENTAL DOCUMENT").upper(), doc_index),
        _title_flowable(title, title_mode),
        _letterhead(
            shared.producer.agency if shared.producer else shared.carrier_name,
            tagline="Underwriting / Loss Control",
            address=(shared.producer.address if shared.producer else None),
        ),
        Spacer(1, 5),
    ]
    if doc.unrelated_subtitle:
        story += [Paragraph(_sanitize(doc.unrelated_subtitle), _SUB), Spacer(1, 4)]
    if doc.nr_header_rows:
        story += [_schedule_table(doc.nr_header_rows), Spacer(1, 6)]
    for block in (doc.unrelated_body or "").split("\n\n"):
        if block.strip():
            story.append(Paragraph(_sanitize(block).replace("\n", "<br/>"), _BODY))
    if doc.nr_columns:
        story += [Spacer(1, 2), _section("SUPPORTING DETAIL"), _data_table(doc.nr_columns, doc.nr_rows)]
    story += [
        Spacer(1, 6),
        Paragraph("Provided for informational and underwriting review purposes only.", _SUB),
    ]
    return story


_DISPATCH = {
    DC.CERTIFICATE: render_certificate,
    DC.INVOICE: render_invoice,
    DC.DECLARATIONS: render_declarations,
    DC.ENDORSEMENT: render_endorsement,
    DC.NEEDS_REVIEW: render_needs_review,
}


# --------------------------------------------------------------------------- #
# Boilerplate constants (exact ACORD language)
# --------------------------------------------------------------------------- #
_ACORD_DISCLAIMER = (
    "THIS CERTIFICATE IS ISSUED AS A MATTER OF INFORMATION ONLY AND CONFERS NO RIGHTS UPON THE "
    "CERTIFICATE HOLDER. THIS CERTIFICATE DOES NOT AFFIRMATIVELY OR NEGATIVELY AMEND, EXTEND OR "
    "ALTER THE COVERAGE AFFORDED BY THE POLICIES BELOW. THIS CERTIFICATE OF INSURANCE DOES NOT "
    "CONSTITUTE A CONTRACT BETWEEN THE ISSUING INSURER(S), AUTHORIZED REPRESENTATIVE OR PRODUCER, "
    "AND THE CERTIFICATE HOLDER."
)
_ACORD_IMPORTANT = (
    "IMPORTANT: If the certificate holder is an ADDITIONAL INSURED, the policy(ies) must have "
    "ADDITIONAL INSURED provisions or be endorsed. If SUBROGATION IS WAIVED, subject to the terms "
    "and conditions of the policy, certain policies may require an endorsement. A statement on this "
    "certificate does not confer rights to the certificate holder in lieu of such endorsement(s)."
)
_CANCELLATION_TEXT = (
    "SHOULD ANY OF THE ABOVE DESCRIBED POLICIES BE CANCELLED BEFORE THE EXPIRATION DATE THEREOF, "
    "NOTICE WILL BE DELIVERED IN ACCORDANCE WITH THE POLICY PROVISIONS."
)


def build_pdf(
    shared: SharedFacts,
    typed_docs: list[tuple[DocumentCategory, DocumentContent]],
    *,
    doc_modes: list[str] | None = None,
    watermarks: list[str | None] | None = None,
    layouts: list[int] | None = None,
) -> bytes:
    """Render one PDF containing every ``(doc_type, content)`` in order.

    Per-document realism knobs (all default to today's behavior when omitted):
      * ``doc_modes[i]`` — title/boundary mode (see ``_title_flowable``); default
        ``"hash"`` (the splittable ``# TITLE``).
      * ``watermarks[i]`` — diagonal DRAFT/COPY/… stamp, or ``None``.
      * ``layouts[i]`` — layout variant int (section order / letterhead tagline).
    """
    story: list = []
    for i, (doc_type, doc) in enumerate(typed_docs):
        mode = doc_modes[i] if doc_modes else "hash"
        wm = watermarks[i] if watermarks else None
        layout = layouts[i] if layouts else 0
        story.append(_WatermarkCue(wm))  # set/clear the watermark for this doc's pages
        story += _DISPATCH[doc_type](shared, doc, doc_index=i, title_mode=mode, layout=layout)
        if i != len(typed_docs) - 1:
            story.append(PageBreak())

    buf = io.BytesIO()
    SimpleDocTemplate(
        buf,
        pagesize=letter,
        title="Insurance Document Packet",
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.55 * inch,
    ).build(story, canvasmaker=NumberedCanvas)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Deferred track: non-PDF and scanned (image-only) single-document renderers.
# Off by default; enabled per-doc via realism.DocNoise.render_format. Heavy deps
# (pdfminer, Pillow) are imported lazily so the default PDF path stays light.
# --------------------------------------------------------------------------- #
def _single_doc_text(shared: SharedFacts, doc_type: DocumentCategory, doc: DocumentContent) -> str:
    """The plain text of a single document (as pdfminer would extract it) — reused as
    the source for both the HTML and the scanned-image renderers so content stays
    coherent with the PDF path."""
    pdf = build_pdf(shared, [(doc_type, doc)], doc_modes=["plain"])
    try:
        from pdfminer.high_level import extract_text

        return extract_text(io.BytesIO(pdf)) or (doc.title or "Insurance Document")
    except Exception:  # noqa: BLE001 — never let a scan/html render kill a run
        return doc.title or "Insurance Document"


def build_html(shared: SharedFacts, doc_type: DocumentCategory, doc: DocumentContent) -> bytes:
    """A single document as an HTML attachment. Still text-extractable (MarkItDown
    parses HTML), so it classifies normally — it just exercises a non-PDF format."""
    title = _esc(doc.title or "Insurance Document")
    paras = "".join(f"<p>{_esc(ln)}</p>\n" for ln in _single_doc_text(shared, doc_type, doc).splitlines() if ln.strip())
    # <h2> not <h1>: avoids MarkItDown emitting a single-'#' boundary line.
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title></head><body><h2>{title}</h2>\n{paras}</body></html>"
    )
    return html.encode("utf-8")


def build_scanned_pdf(shared: SharedFacts, doc_type: DocumentCategory, doc: DocumentContent) -> bytes:
    """A single document as an image-only PDF (no text layer), simulating a scanned or
    faxed submission. pdfminer extracts nothing from it, so the graph — which has no
    OCR — routes it to needs_review. That is the intended honest signal."""
    from PIL import Image, ImageDraw, ImageFont

    text = "[SCANNED COPY]\n\n" + _single_doc_text(shared, doc_type, doc)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:  # noqa: BLE001 — DejaVu not always present; bitmap default still has no text layer
        font = ImageFont.load_default()

    W, H, margin, line_h, max_chars = 850, 1100, 60, 15, 95
    lines: list[str] = []
    for raw in text.splitlines():
        raw = raw.rstrip()
        while len(raw) > max_chars:
            lines.append(raw[:max_chars])
            raw = raw[max_chars:]
        lines.append(raw)

    per_page = max(1, (H - 2 * margin) // line_h)
    pages = []
    for start in range(0, max(1, len(lines)), per_page):
        img = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(img)
        y = margin
        for ln in lines[start:start + per_page]:
            draw.text((margin, y), ln, fill=(25, 25, 25), font=font)
            y += line_h
        pages.append(img)
    if not pages:
        pages = [Image.new("RGB", (W, H), "white")]

    buf = io.BytesIO()
    pages[0].save(buf, format="PDF", save_all=True, append_images=pages[1:])
    return buf.getvalue()
