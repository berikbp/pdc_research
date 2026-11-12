"""Build supervisor slides, handout and notes from the versioned result tables.

Optional document dependencies are separate from benchmark dependencies.
See docs/presentation/README.md for the build command.
"""
from pathlib import Path
import csv
import json
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Pt
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/presentation'
FIG = OUT / 'figures'
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')
INK, MUTED, TEAL, ORANGE, LIGHT, WHITE = '#16283C', '#526477', '#087F8C', '#CF6B32', '#EFF4F7', '#FFFFFF'
LABELS = {
    'gpu-model-only': 'GPU-only model graph', 'g99-t1': 'Full offload / CPU emb. / 1 thread',
    'g99-t4': 'Full offload / CPU emb. / 4 threads', 'g99-t8': 'Full offload / CPU emb. / 8 threads',
    'g99-t16': 'Full offload / CPU emb. / 16 threads', 'g28-t4': '28 GPU layers / 4 threads',
    'g28-t8': '28 GPU layers / 8 threads', 'g24-t4': '24 GPU layers / 4 threads',
    'g24-t8': '24 GPU layers / 8 threads',
}


def read_csv(name):
    with (ROOT / name).open() as stream:
        return list(csv.DictReader(stream))


def figures(controlled, languages):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    ids = [r['configuration'] for r in controlled]
    means = [float(r['mean_total_s']) for r in controlled]
    ax.barh(range(9), means, color=[TEAL if i < 5 else '#9CADBA' for i in range(9)], height=.65)
    for i, row in enumerate(controlled):
        ax.scatter([float(row[f'round_{j}_s']) for j in (1,2,3)], [i]*3,
                   marker='|', s=115, color=INK, zorder=3)
        ax.text(means[i]+.06, i, f'{means[i]:.3f} s', va='center', fontsize=10)
    ax.set_yticks(range(9), [LABELS[c] for c in ids]); ax.invert_yaxis()
    ax.set_xlim(0, 3.1); ax.set_xlabel('Prompt processing + 64-token generation (seconds; lower is better)')
    ax.xaxis.grid(True, alpha=.15); ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ('png','svg'): fig.savefig(FIG/f'controlled-configurations.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    names = ['English', 'Russian', 'Kazakh', 'Chinese']
    codes = ['eng_Latn','rus_Cyrl','kaz_Cyrl','zho_Hans']
    values = [float(next(x['mean_total_s'] for x in languages if x['configuration']=='gpu-model-only' and x['language']==code)) for code in codes]
    fig, ax = plt.subplots(figsize=(8.5,3.5))
    ax.bar(names, values, color=TEAL, width=.55)
    for i, value in enumerate(values): ax.text(i, value+.04, f'{value:.4f} s', ha='center')
    ax.set_ylim(0,1.8); ax.set_ylabel('Inference time (seconds)')
    ax.yaxis.grid(True,alpha=.15); ax.set_axisbelow(True); fig.tight_layout()
    for ext in ('png','svg'): fig.savefig(FIG/f'controlled-languages.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


class Deck:
    W, H = 960, 540
    def __init__(self):
        self.ppt = Presentation()
        self.ppt.slide_width, self.ppt.slide_height = Pt(self.W), Pt(self.H)
        self.ppt.core_properties.title = 'Multilingual CPU–GPU inference: research progress'
        self.ppt.core_properties.author = 'PDC Research Team'
        self.pdf = canvas.Canvas(str(OUT/'supervisor-presentation.pdf'), pagesize=(self.W,self.H))
        self.pdf.setTitle('Multilingual CPU–GPU inference: research progress')
        self.number, self.notes = 0, []

    def rect(self,x,y,w,h,fill):
        shape=self.slide.shapes.add_shape(1,Pt(x),Pt(y),Pt(w),Pt(h))
        shape.fill.solid();shape.fill.fore_color.rgb=RGBColor.from_string(fill[1:]);shape.line.fill.background()
        self.pdf.setFillColor(colors.HexColor(fill));self.pdf.rect(x,self.H-y-h,w,h,fill=1,stroke=0)

    def text(self,x,y,w,h,value,size=22,color=INK,bold=False):
        font='DejaVuSans-Bold' if bold else 'DejaVuSans'
        lines=[]
        for paragraph in str(value).split('\n'):
            words=paragraph.split();line=''
            for word in words:
                candidate=(line+' '+word).strip()
                if pdfmetrics.stringWidth(candidate,font,size)>w and line:
                    lines.append(line);line=word
                else:line=candidate
            lines.append(line)
        if len(lines)*size*1.18>h+1:
            raise ValueError(f'Slide {self.number}: text overflows its box: {value[:60]}')
        tf=self.slide.shapes.add_textbox(Pt(x),Pt(y),Pt(w),Pt(h)).text_frame
        tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0;tf.word_wrap=False
        self.pdf.setFillColor(colors.HexColor(color));self.pdf.setFont(font,size)
        for i,line in enumerate(lines):
            p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.line_spacing=1.18;p.space_after=Pt(0)
            for run in p.runs:
                run.font.name='DejaVu Sans';run.font.size=Pt(size);run.font.bold=bold;run.font.color.rgb=RGBColor.from_string(color[1:])
            self.pdf.drawString(x,self.H-y-size*.93-i*size*1.18,line)

    def start(self,title,section,note,source='docs/current-report.md'):
        if self.number:self.pdf.showPage()
        self.number+=1;self.slide=self.ppt.slides.add_slide(self.ppt.slide_layouts[6])
        self.rect(0,0,self.W,self.H,WHITE);self.rect(0,0,12,self.H,TEAL)
        self.text(40,26,870,20,section.upper(),12,TEAL,True)
        self.text(40,56,875,78,title,30,INK,True)
        self.rect(40,502,880,1,'#DCE4EA')
        self.text(40,515,820,15,f'PDC research | 26 October 2026 | Source: {source}',9,MUTED)
        self.text(892,513,30,18,str(self.number),11,MUTED)
        self.slide.notes_slide.notes_text_frame.text=note
        self.notes.append((self.number,title,note))

    def bullets(self,items,x=50,y=145,w=850,size=22,gap=22):
        for item in items:
            # Allow up to two wrapped lines per point at ordinary body size.
            self.rect(x,y+7,6,6,TEAL)
            height=55 if size>=20 else 48
            self.text(x+20,y,w-20,height,item,size)
            y+=height+gap

    def table(self,headers,rows,x=40,y=140,widths=None,size=15,row_h=31):
        widths=widths or [880/len(headers)]*len(headers)
        for i,row in enumerate([headers]+rows):
            self.rect(x,y+i*row_h,sum(widths),row_h,INK if i==0 else (LIGHT if i%2 else WHITE))
            xx=x
            for value,width in zip(row,widths):
                self.text(xx+9,y+i*row_h+6,width-16,row_h-8,str(value),size,WHITE if i==0 else INK,i==0)
                xx+=width

    def image(self,path,x,y,w,h):
        self.slide.shapes.add_picture(str(path),Pt(x),Pt(y),width=Pt(w),height=Pt(h))
        self.pdf.drawImage(str(path),x,self.H-y-h,width=w,height=h,mask='auto')

    def save(self):
        self.ppt.save(OUT/'supervisor-presentation.pptx');self.pdf.save()
        content=['# Speaker notes and discussion guide','','Suggested delivery: 8–10 minutes for slides 1–11. Slides 12–14 are backup tables.','']
        for n,title,note in self.notes:content.extend([f'## Slide {n}: {title}','',note,''])
        content += ['## Suggested opening statement','',
                    'We tested whether language changes which CPU–GPU inference configuration is fastest beyond token length. Our initial measurements mixed language with input size. We corrected that with a controlled 64-token experiment. In that restricted setting, we found no stable, practically meaningful language-dependent configuration preference. We would like guidance on replication and extending the workload before evaluating a scheduling policy.','',
                    '## Questions the supervisor may ask','',
                    '**Is this a universal negative result?** No. It covers one model, laptop, short input/output length, a curated dev sample and three rounds within one session. Fifteen interaction intervals were inconclusive for the practical-equivalence criterion.','',
                    '**Did CPU and GPU cooperate?** Layer placement and CPU thread settings were varied. This does not establish an asynchronous cooperative backend. Full offload with CPU embedding was distinguished from the explicit CUDA-embedding candidate.','',
                    '**Why are semantically aligned passages not enough?** Token counts differ; the controlled prefixes equalize computation length but lose exact semantic equivalence. Both experiments have distinct roles.','',
                    '**What is novel?** Novelty has not been established. The current contribution is reproducible characterization and a bounded negative finding, pending replication and comparison with prior work.','',
                    '**What about RQ2?** No selector was implemented or evaluated. The current setting shows no demonstrated language-specific benefit that would justify one.','',
                    '**Does one thread mean zero CPU use?** No. Parameter one avoids additional OpenMP inference workers; CPU orchestration, sampling and other host work remain allowed. The diagnostic verified model graph placement for a short generation, not every measured request.','',
                    '**Should the study stop?** Discuss whether to replicate and extend the original question or formally revise the project toward memory-constrained CPU–GPU cooperation. They require different experiments.','']
        (OUT/'speaker-notes.md').write_text('\n'.join(content))


def handout():
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyCustom',fontName='DejaVuSans',fontSize=9.2,leading=11.5,spaceAfter=5,textColor=colors.HexColor(INK)))
    styles.add(ParagraphStyle(name='TitleCustom',fontName='DejaVuSans-Bold',fontSize=18,leading=21,spaceAfter=8,textColor=colors.HexColor(INK)))
    styles.add(ParagraphStyle(name='HeadingCustom',fontName='DejaVuSans-Bold',fontSize=10.5,leading=13,spaceBefore=5,spaceAfter=4,textColor=colors.HexColor(TEAL)))
    blocks=[('title','Multilingual CPU–GPU inference: progress summary'),
            ('body','Supervisor discussion | 26 October 2026 | Initial-machine results'),
            ('heading','Question and current answer'),
            ('body','RQ1 asks whether language changes the best CPU–GPU configuration after controlling tokenized sequence length. In the controlled 64-input / 64-output-token experiment, we found no stable, practically meaningful language-dependent configuration preference among nine tested settings. This is a restricted negative finding; RQ2 (policy improvement) remains unanswered.'),
            ('heading','What was done'),
            ('body','Stage 1: 960 aligned-content trials across 12 thread/offload settings. Full layer offload won all 80 input comparisons, but token counts differed and there was one repetition. Kazakh averaged 298.8 input tokens versus 92.8 for English.'),
            ('body','Stage 2: 2,160 trials covering the CPU-thread sweep, ten-thread follow-up and GPU-only model baseline. This was useful hardware/backend characterization but partly a detour from RQ1. More CPU threads did not consistently help. A diagnostic found CPU embedding computation under default full offload; explicitly moving the embedding to CUDA corrected the GPU-only model baseline.'),
            ('body','Stage 3: 2,052 trials = 19 eligible passage groups × four languages × nine configurations × three shuffled rounds. Every trial used 64 input and 64 output tokens. All records passed validation; zero error records. Total across all stages: 5,172 measured requests, excluding warmups and diagnostics.'),
            ('heading','Controlled measurements'),
            ('table',None),
            ('body','Means include prompt processing and generation; loading and warmups are excluded. GPU-only model computation permits CPU orchestration/sampling. The GPU-only baseline averaged 1.434 s English, 1.433 s Russian, 1.434 s Kazakh and 1.432 s Chinese.'),
            ('heading','Uncertainty and limits'),
            ('body','10,000 passage-cluster bootstrap resamples; approximate simultaneous 95% intervals. Of 216 language-by-configuration interactions, 201 intervals lay within the predeclared ±2-percentage-point practical band; 15 were inconclusive. One small interaction excluded zero but stayed below the practical threshold. None met the stable, practically meaningful criterion.'),
            ('body','One laptop (Ryzen 7 7435HS / RTX 4060 8 GiB, WSL), one model (Qwen2.5-7B-Instruct Q4_K_M), one short length, curated dev passages and rounds within one session. Prefix clipping loses semantic equivalence. No power/utilization telemetry, held-out policy evaluation, universal equivalence or unique CPU-core optimum is established.'),
            ('heading','Decision requested from the supervisor'),
            ('body','Recommended: replicate the controlled protocol on teammates’ machines, then test longer contexts if warranted. Alternatively, formally revise the question toward memory-constrained CPU–GPU cooperation. Do not build a language-aware selector unless replicated measurements show a useful signal.'),
            ('body','Evidence: docs/current-report.md and docs/results/rq1-controlled-64-v1/. Repository: github.com/berikbp/pdc_research.')]
    story=[];md=['# Supervisor progress summary','']
    for kind,value in blocks:
        if kind=='table':
            data=[['Configuration','Mean inference'],['GPU-only model graph','1.433 s'],['Full offload / CPU embedding / 4 threads','1.434 s'],['28 GPU layers / 4 threads','1.677 s'],['24 GPU layers / 8 threads','2.494 s']]
            table=Table(data,colWidths=[360,125],rowHeights=[19]*5)
            table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'DejaVuSans'),('FONTSIZE',(0,0),(-1,-1),9),('BACKGROUND',(0,0),(-1,0),colors.HexColor(INK)),('TEXTCOLOR',(0,0),(-1,0),colors.white),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor(LIGHT),colors.white]),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
            story.extend([table,Spacer(1,6)])
            md += ['| Configuration | Mean inference |','|---|---:|',*[f'| {a} | {b} |' for a,b in data[1:]],'']
        else:
            sty={'title':'TitleCustom','heading':'HeadingCustom','body':'BodyCustom'}[kind]
            story.append(Paragraph(value,styles[sty]));md.extend([('## ' if kind=='heading' else '')+value,''])
    SimpleDocTemplate(str(OUT/'supervisor-summary.pdf'),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=32,bottomMargin=28).build(story)
    (OUT/'supervisor-summary.md').write_text('\n'.join(md))


def main():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)
    for name,file in [('DejaVuSans','DejaVuSans.ttf'),('DejaVuSans-Bold','DejaVuSans-Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(FONT_DIR/file)))
    controlled=read_csv('docs/results/rq1-controlled-64-v1/configurations.csv')
    languages=read_csv('docs/results/rq1-controlled-64-v1/languages.csv')
    history=read_csv('docs/results/history/exploratory-configurations.csv')
    validation=json.loads((ROOT/'docs/results/rq1-controlled-64-v1/validation.json').read_text())
    assert sum(int(r['n']) for r in controlled)==2052 and validation['validated_trials']==2052
    figures(controlled,languages)
    d=Deck()
    d.start('Does language change the best\nCPU–GPU inference configuration?', 'Research progress / initial-machine results',
            'Open with the question, not a claim of universal GPU superiority. The short answer is a restricted negative finding at equal token counts. State that this is a progress review and request guidance before expanding the study.')
    d.text(48,160,850,100,'No stable, practically meaningful language-dependent preference detected.',30,TEAL,True)
    d.text(48,286,850,65,'One model · one laptop · 64 input / 64 output tokens',21)
    d.rect(42,386,878,80,LIGHT)
    for x,big,label in [(58,'5,172','measured requests overall'),(355,'2,052','controlled RQ1 requests'),(650,'3','shuffled controlled rounds')]:
        d.text(x,395,260,33,big,26,TEAL,True);d.text(x,437,260,20,label,13,MUTED)
    d.start('RQ1 asks whether the winner changes by language','01 / Research question',
            'RQ1 is not simply which configuration is fastest or which language takes longer. It is whether language changes configuration preferences after controlling token count. RQ2 is conditional: a useful selector requires a reproducible signal and held-out evaluation.')
    d.bullets(['RQ1: at equal token counts, do languages favor different CPU–GPU configurations?',
               'RQ2: if a useful difference exists, can a workload-aware policy beat fixed and length-aware policies?',
               'Different latency alone is insufficient: the configuration preference must change meaningfully.'],size=21,gap=35)
    d.start('A reproducible benchmark on one consumer laptop','02 / Experimental setting',
            'All results presented were collected on the initial laptop. Other team machines have not yet supplied a replicated controlled result. The FLORES passages are curated, not randomly representative. This is performance measurement, not an accuracy benchmark.')
    d.bullets(['Ryzen 7 7435HS: 8 physical cores / 16 logical CPUs; RTX 4060 Laptop: ~8 GiB VRAM; WSL.',
               'Qwen2.5-7B-Instruct Q4_K_M; pinned llama.cpp; FLORES-200 English, Russian, Kazakh and Chinese.',
               'Primary metric: prompt-processing + generation latency. Loading and warmups excluded.'],size=20,gap=34)
    d.start('Three stages, with a correction in direction','03 / Research history',
            'Be candid: stage 1 was useful exploration. Stage 2 partly shifted away from RQ1 into thread tuning and baseline clarification. Stage 3 returned to the required equal-length comparison. Counts exclude setup, warmups and diagnostic runs.')
    d.table(['Stage','Measured trials','What it established'],[
        ['1. Aligned-content pilot','960','Offload comparison; length confounded'],
        ['2. Threads + baseline','2,160','Supporting characterization; partial detour'],
        ['3. Controlled RQ1','2,052','Equal-length, repeated comparison']], widths=[290,150,440],row_h=51,size=17)
    d.text(50,380,850,70,'The second stage improved backend understanding, but did not directly answer language-dependent configuration selection.',21,TEAL,True)
    d.start('Stage 1: input size differed substantially by language','04 / Exploratory pilot',
            'Full layer offload won all 80 inputs among the tested placements, but each condition had one repetition. Kazakh inputs averaged about 3.2 times the tokens of English for aligned content. This is consistent with workload-size differences; it does not isolate a language effect or prove novelty.')
    d.table(['Language','Mean input tokens','Full offload / 4 threads'],[
        ['English','92.8','1.470 s'],['Russian','168.2','1.513 s'],['Kazakh','298.8','1.603 s'],['Chinese','93.0','1.468 s']],widths=[260,260,360],row_h=43,size=18)
    d.text(50,390,850,64,'Aligned meaning ≠ equal computation length. The pilot could not isolate language from token count.',22,TEAL,True)
    d.start('Stage 2: thread tuning was not the central RQ1 test','05 / Detour and baseline correction',
            'The original full-offload thread sweep gave 1.4744 seconds with one thread and 1.4762 with twelve. Those tiny differences and session/block confounds do not identify a definitive optimum. We found default full layer offload still used CPU for embedding lookup. Explicit CUDA embedding made all logged model graph nodes CUDA in a short diagnostic. CPU orchestration and sampling remain allowed. Thread counts are not physical-core allocation.')
    d.bullets(['More CPU threads did not consistently improve full-offload performance; a unique optimum was not established.',
               'Full layer offload still used CPU embedding lookup in the diagnostic. Explicit CUDA embedding corrected the model-graph baseline.',
               'This was useful characterization, but unequal token counts and blocked repetitions left RQ1 unresolved.'],size=20,gap=34)
    d.start('Stage 3 equalized the workload and revisited settings','06 / Controlled design',
            'Exactly 64 input and 64 output tokens. Nineteen of twenty groups were eligible in every language; the excluded group had 56 Chinese tokens. Token prefixes lose exact semantic equivalence. Three rounds revisited every configuration in shuffled order, with requests shuffled within blocks. All 2052 records passed validation with no error records. The 2 percent practical band was specified before execution.')
    d.text(50,143,850,45,'19 groups × 4 languages × 9 settings × 3 rounds = 2,052',25,TEAL,True)
    d.bullets(['GPU-only model graph; full layer offload with CPU embedding at 1 / 4 / 8 / 16 threads.',
               'Partial offload: 28 or 24 GPU layers, each with 4 or 8 CPU threads.',
               'All records validated: 64 input / 64 output; no cache reuse, truncation or recorded errors.'],y=220,size=20,gap=20)
    d.start('GPU-dominant settings formed the fastest group','07 / Controlled configuration results',
            'The bars show overall means; ticks are means from the three observed rounds, not confidence intervals. Each overall value averages 228 measurements. GPU-only averaged 1.4330 seconds and full offload with four CPU threads 1.4337: roughly 0.05 percent apart. Do not claim a unique winner. Partial offload was clearly slower in this sample. The 24-layer four-thread setting also shifted across rounds.', 'docs/results/.../configurations.csv')
    d.image(FIG/'controlled-configurations.png',28,135,895,330)
    d.text(48,473,860,22,'Bars: overall means. Ticks: three round means, not confidence intervals.',13,MUTED)
    d.start('Equal-length inputs had very similar GPU-only timings','08 / Language comparison',
            'These are means under the explicit CUDA-embedding GPU-only model candidate. The vertical axis begins at zero. Configuration winner changes among the full-offload settings were numerically tiny. Raw similarity alone is not the full RQ1 analysis; the next slide discusses the language-by-configuration contrasts and uncertainty.', 'docs/results/.../languages.csv')
    d.image(FIG/'controlled-languages.png',100,130,750,308)
    d.text(50,452,850,45,'Similar means are descriptive evidence; uncertainty determines how strong the conclusion can be.',16,TEAL,True)
    d.start('No interaction met the practical, stable-effect criteria','09 / Uncertainty and RQ1 answer',
            'Ten thousand bootstrap samples resampled the nineteen passage groups, retaining languages, settings and rounds together. Intervals were adjusted within the complete family of 216 interactions. 201 intervals were within plus/minus two percentage points. Fifteen remained inconclusive for equivalence. One small interaction excluded zero, estimated at 0.655 points, but remained below the practical threshold. No interaction met all criteria. These intervals are conditional on observed rounds and do not estimate future-session variability.', 'docs/results/.../validation.json')
    for x,big,small in [(50,'201 / 216','intervals within ±2 points'),(355,'15 / 216','inconclusive for equivalence'),(660,'0','stable practical interactions')]:
        d.rect(x-8,147,286,112,LIGHT);d.text(x,163,268,43,big,31,TEAL,True);d.text(x,220,270,30,small,14,MUTED)
    d.text(50,297,850,62,'Restricted negative finding: no useful language-dependent configuration preference detected here.',25,TEAL,True)
    d.text(50,380,850,100,'One small interaction was statistically detectable but below the practical threshold. Scope: one model, one laptop, one short length. RQ2 remains unanswered.',18)
    d.start('Supervisor decision: replicate, extend, or revise?','10 / Proposed next step',
            'Recommend replication of the controlled protocol on teammates hardware, followed by longer contexts if warranted. Keep machines separate in analysis. If the supervisor instead wants a study of memory-constrained cooperative CPU/GPU computation, revise the proposal explicitly and select an appropriate model/backend. Do not build a language-aware selector merely because RQ2 was proposed. Ask which direction best fits the project scope.')
    d.bullets(['Recommended: reproduce the controlled protocol on teammates’ machines, then consider longer contexts.',
               'Alternative: formally revise the study toward memory-constrained CPU–GPU cooperation.',
               'RQ2 remains unanswered. Build a selector only if replicated data demonstrates a useful signal.'],size=21,gap=32)
    d.start('Backup: controlled rounds side by side','Appendix A / Complete controlled matrix',
            'Each cell averages 76 inputs; every input has 64 prompt and 64 generated tokens. A round is a revisit to all nine settings, within one session. Full means, source data and all uncertainty tables are versioned. GPU-only graph placement was verified in a short diagnostic rather than traced during every timing trial.', 'docs/results/.../configurations.csv')
    rows=[[r['configuration'],*[f"{float(r[k]):.4f}" for k in ['round_1_s','round_2_s','round_3_s','mean_total_s']]] for r in controlled]
    d.table(['Configuration','Round 1','Round 2','Round 3','Mean'],rows,widths=[300,145,145,145,145],row_h=32,size=15)
    d.text(50,475,850,20,'Seconds; loading/warmups excluded. All g99 entries use CPU embedding except gpu-model-only.',12,MUTED)
    d.start('Backup: the original 12-setting pilot','Appendix B / Stage 1',
            'Each cell averages 80 aligned-content inputs, with one repetition. The offload setting refers to model layers, not GPU cores. Input lengths differ across languages. These numbers should not be pooled with the controlled-prefix run as if they were the same workload.', 'docs/results/history/exploratory-configurations.csv')
    rows=[]
    for g in [0,8,16,24,28,99]:
        values=[next(r for r in history if r['run']=='berik-pilot-v3' and int(r['threads'])==t and int(r['gpu_layers'])==g) for t in [4,8]]
        rows.append(['All requested' if g==99 else str(g),*[f"{float(r['mean_total_s']):.4f}" for r in values]])
    d.table(['GPU layers','4 CPU threads, s','8 CPU threads, s'],rows,widths=[300,290,290],row_h=40,size=18)
    d.start('Backup: exploratory thread and baseline repetitions','Appendix C / Stage 2',
            'These are blocked repetitions with unequal input lengths. The ten-thread and CUDA-embedding baseline rows are follow-up sessions. The table documents the investigation but must not be used to infer a causal advantage from tiny cross-session differences.', 'docs/results/history/exploratory-configurations.csv')
    rows=[]
    for t in [1,2,4,6,8,10,12,16]:
        run='berik-cpu-threads-10-v1' if t==10 else 'berik-cpu-threads-v1'
        r=next(r for r in history if r['run']==run and int(r['threads'])==t)
        rows.append([f'CPU emb. / {t} threads',*[f"{float(r[f'repetition_{i}_s']):.4f}" for i in [1,2,3]],f"{float(r['mean_total_s']):.4f}"])
    r=next(r for r in history if r['run']=='berik-gpu-model-baseline-v1')
    rows.append(['CUDA emb. / 1 thread',*[f"{float(r[f'repetition_{i}_s']):.4f}" for i in [1,2,3]],f"{float(r['mean_total_s']):.4f}"])
    d.table(['Placement / threads','Rep. 1','Rep. 2','Rep. 3','Mean'],rows,widths=[300,145,145,145,145],row_h=32,size=15)
    d.text(50,475,850,20,'Seconds. Repetitions within a configuration block are not independent experiment sessions.',12,MUTED)
    d.save();handout()
    print(f'Created {d.number} slides, PDF handout, speaker notes and standalone figures in {OUT}')


if __name__ == '__main__':
    main()
