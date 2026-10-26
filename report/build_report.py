"""Build the coursework PDF from the reviewed Markdown source with ReportLab."""

import argparse
import html
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Preformatted, Table, TableStyle)

NAVY=colors.HexColor('#19364D')
INK=colors.HexColor('#17212B')
PALE=colors.HexColor('#EFF3F6')
MUTED=colors.HexColor('#50616D')


def inline(text):
    result=html.escape(text)
    result=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',result)
    result=re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',result)
    result=re.sub(r'(https?://[^\s<]+)',r'<link href="\1" color="#174E78">\1</link>',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path(__file__).with_name('report.md'))
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('MP2_Report.pdf'))
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyReport',fontName='Times-Roman',fontSize=10.5,
        leading=13.5,textColor=INK,spaceAfter=7,alignment=TA_LEFT,allowWidows=0,allowOrphans=0))
    styles.add(ParagraphStyle(name='TitleReport',fontName='Helvetica-Bold',fontSize=20,
        leading=23,textColor=NAVY,spaceAfter=10))
    styles.add(ParagraphStyle(name='SectionReport',fontName='Helvetica-Bold',fontSize=12.2,
        leading=15,textColor=NAVY,spaceBefore=6,spaceAfter=8,keepWithNext=True))
    styles.add(ParagraphStyle(name='SubsectionReport',fontName='Helvetica-Bold',fontSize=10.2,
        leading=13,textColor=NAVY,spaceBefore=5,spaceAfter=6,keepWithNext=True))
    styles.add(ParagraphStyle(name='CellReport',fontName='Helvetica',fontSize=8.6,
        leading=10.9,textColor=INK,spaceAfter=0))
    styles.add(ParagraphStyle(name='HeadCellReport',parent=styles['CellReport'],fontName='Helvetica-Bold',textColor=colors.white))
    styles.add(ParagraphStyle(name='ReferenceReport',fontName='Times-Roman',fontSize=8.5,
        leading=10.6,textColor=INK,spaceAfter=6,splitLongWords=True))
    styles.add(ParagraphStyle(name='CodeReport',fontName='Courier',fontSize=8.2,
        leading=11,textColor=INK,borderWidth=.4,borderColor=colors.HexColor('#C9D5DE'),
        borderPadding=7,backColor=PALE,spaceBefore=5,spaceAfter=10))
    width=A4[0]-104
    fractions=[(.46,.18,.36),(.21,.79),(.46,.19,.15,.20),(.52,.16,.16,.16),
        (.43,.15,.42),(.43,.30,.27),(.44,.56)]
    lines=args.source.read_text(encoding='utf-8').splitlines()
    story=[];i=0;table_index=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1;continue
        if line=='<!-- PAGEBREAK -->':
            story.append(PageBreak());i+=1;continue
        if line.startswith('```'):
            i+=1;block=[]
            while i<len(lines) and not lines[i].strip().startswith('```'):
                # Preserve command line breaks and explicit shell continuations.
                block.append(lines[i])
                i+=1
            story.append(Preformatted('\n'.join(block),styles['CodeReport']));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[s.strip() for s in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',s) for s in cells):rows.append(cells)
                i+=1
            ratios=fractions[table_index];table_index+=1
            assert len(ratios)==len(rows[0])
            data=[[Paragraph(inline(c),styles['HeadCellReport'] if r==0 else styles['CellReport'])
                for c in row] for r,row in enumerate(rows)]
            table=Table(data,colWidths=[width*x for x in ratios],repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
                ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),
                ('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#C6D2DA'))]))
            story.extend([table,Spacer(1,9)]);continue
        if line.startswith('# '):
            story.append(Paragraph(inline(line[2:]),styles['TitleReport']));i+=1;continue
        if line.startswith('## '):
            story.append(Paragraph(inline(line[3:]),styles['SectionReport']));i+=1;continue
        if line.startswith('### '):
            story.append(Paragraph(inline(line[4:]),styles['SubsectionReport']));i+=1;continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#','|','```','<!-- PAGEBREAK')):
            paragraph.append(lines[i].strip());i+=1
        text=' '.join(paragraph)
        style=styles['ReferenceReport'] if re.match(r'^\[\d+\]',text) else styles['BodyReport']
        story.append(Paragraph(inline(text),style))
    assert table_index==7
    def decorate(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#C5D0D8'));canvas.setLineWidth(.4)
        canvas.line(52,A4[1]-32,A4[0]-52,A4[1]-32)
        canvas.setFont('Helvetica',7.5);canvas.setFillColor(MUTED)
        canvas.drawString(52,A4[1]-24,'DASE7506 | Mini-project 2 | Wu Houlin')
        canvas.drawString(52,25,'3036863317 | Revised 27 October 2026')
        canvas.drawRightString(A4[0]-52,25,f'{doc.page}')
        canvas.restoreState()
    doc=SimpleDocTemplate(str(args.output),pagesize=A4,rightMargin=52,leftMargin=52,
        topMargin=47,bottomMargin=43,title='Selecting Reliable LLM Reasoning',author='Wu Houlin',
        subject='DASE7506 Mini-project 2, student3036863317',creator='ReportLab; Codex-assisted coursework report')
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    print(args.output.resolve())


if __name__=='__main__':main()
