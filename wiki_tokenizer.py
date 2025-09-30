import argparse
import bz2
import html
import json
import re
import sys
import time
from pathlib import Path
import xml.etree.ElementTree as ET

from ftfy import fix_text
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich import box

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DUMP = BASE_DIR / "data" / "idwiki-20250501-pages-articles-multistream.xml.bz2"
DEFAULT_OUT  = BASE_DIR / "outputs" / "wiki-mini"
DEFAULT_PREFIX = str(BASE_DIR / "outputs" / "veldora_tokenizer")
DEFAULT_VOCAB = 16000
DEFAULT_CHARCOVER = 0.9995
MIN_LINE, MAX_LINE = 20, 2000
UPDATE_EVERY = 200
TITLE_EVERY  = 1000

console = Console()

re_comments   = re.compile(r"<!--.*?-->", re.DOTALL)
re_ref        = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL|re.IGNORECASE)
re_nowiki     = re.compile(r"<nowiki>.*?</nowiki>", re.DOTALL|re.IGNORECASE)
re_disallowed = re.compile(r"</?(math|code|pre|gallery|timeline|source|small|ref|span|div|table|tr|td|th|sup|sub)[^>]*>", re.IGNORECASE)
re_table      = re.compile(r"\{\|.*?\|\}", re.DOTALL)
re_category   = re.compile(r"\[\[Category:[^\]]*\]\]", re.IGNORECASE)
re_file       = re.compile(r"\[\[(File|Image):[^\]]*\]\]", re.IGNORECASE)
re_template   = re.compile(r"\{\{[^{}]*\}\}")
re_extlink    = re.compile(r"\[(?:https?://[^\s\]]+)\s+([^\]]+)\]")
re_heading    = re.compile(r"^=+\s*(.*?)\s*=+$", re.MULTILINE)
re_link_pipe  = re.compile(r"\[\[([^|\]]+)\|([^\]]+)\]\]")
re_link_plain = re.compile(r"\[\[([^\]]+)\]\]")
re_tags       = re.compile(r"<[^>]+>")
re_sentence_split = re.compile(r"(?<=[\.\?\!])\s+|\n+")

def clean_wikitext(text: str) -> str:
    text = html.unescape(text)
    text = fix_text(text)
    text = re_comments.sub(" ", text)
    text = re_ref.sub(" ", text)
    text = re_nowiki.sub(" ", text)
    text = re_table.sub(" ", text)
    text = re_disallowed.sub(" ", text)
    text = re_extlink.sub(r"\1", text)
    text = re_heading.sub(r"\1", text)
    
    for _ in range(3):
        new_text = re_template.sub(" ", text)
        if new_text == text:
            break
        text = new_text
    text = re_file.sub(" ", text)
    text = re_category.sub(" ", text)
    text = re_link_pipe.sub(r"\2", text)
    text = re_link_plain.sub(r"\1", text)
    text = re_tags.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def iter_pages(bz2_path: str):
    with bz2.open(bz2_path, "rb") as f:
        ctx = ET.iterparse(f, events=("end",))
        for ev, elem in ctx:
            if elem.tag.endswith("page"):
                title = elem.findtext(".//{*}title") or ""
                page_id = elem.findtext(".//{*}id") or ""
                text = elem.findtext(".//{*}text") or ""
                yield page_id, title, text
                elem.clear()

def render_status(start_ts, art_count, line_count, last_title, phase="Extract"):
    elapsed = time.time() - start_ts
    speed = art_count / elapsed if elapsed > 0 else 0.0

    table = Table.grid(padding=(0,1))
    table.add_column(justify="right", style="bold cyan")
    table.add_column(style="white")

    table.add_row("🔧 Phase", phase)
    table.add_row("⏱ Elapsed", f"{elapsed:,.1f} s")
    table.add_row("📄 Articles", f"{art_count:,}")
    table.add_row("✍️  Lines", f"{line_count:,}")
    table.add_row("⚡ Speed", f"{speed:,.2f} art/s")
    table.add_row("🧾 Last title", (last_title or "—")[:80])

    panel = Panel(
        table,
        title="[bold magenta]Wiki → Tokenizer Pipeline[/]",
        subtitle="Output: [green]wiki.jsonl[/], [green]data.txt[/], [green].model[/], [green].vocab[/]",
        border_style="magenta",
        box=box.ROUNDED
    )
    return panel

def step_extract(dump_bz2: str, out_dir: Path, sample: int|None = None, dedup: bool=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "wiki.jsonl"
    data_txt = out_dir / "data.txt"

    art_count = 0
    line_count = 0
    last_title = ""
    start_ts = time.time()
    seen_lines = set() if dedup else None

    with Live(render_status(start_ts, art_count, line_count, last_title, "Extract"),
              refresh_per_second=5, console=console) as live:
        try:
            with jsonl.open("w", encoding="utf-8") as jout, data_txt.open("w", encoding="utf-8") as dout:
                for pid, title, raw in iter_pages(dump_bz2):
                    if not raw:
                        continue
                    last_title = title
                    txt = clean_wikitext(raw)
                    if len(txt) < 80:
                        continue

                    jout.write(json.dumps({"id": pid, "title": title, "text": txt}, ensure_ascii=False) + "\n")
                    art_count += 1

                    for seg in re_sentence_split.split(txt):
                        seg = seg.strip()
                        if MIN_LINE <= len(seg) <= MAX_LINE:
                            if seen_lines is not None:
                                h = hash(seg)
                                if h in seen_lines:
                                    continue
                                seen_lines.add(h)
                            dout.write(seg + "\n")
                            line_count += 1

                    if art_count % UPDATE_EVERY == 0:
                        live.update(render_status(start_ts, art_count, line_count, last_title, "Extract"))
                    if art_count % TITLE_EVERY == 0:
                        console.log(f"[dim]Last article[/]: [yellow]{title[:100]}[/]")

                    if sample is not None and art_count >= sample:
                        break

                live.update(render_status(start_ts, art_count, line_count, last_title, "Extract"))

        except KeyboardInterrupt:
            console.print("\n[yellow]Dihentikan oleh user.[/]")
            sys.exit(1)
        except Exception:
            console.print("\n[red]Error saat Extract![/]\n")
            console.print_exception()
            sys.exit(1)

    console.rule("[bold green]Extract selesai")
    console.print(
        Panel.fit(
            f"[green]JSONL:[/] {jsonl}\n"
            f"[green]data.txt:[/] {data_txt}\n"
            f"[cyan]Articles:[/] {art_count:,}   [cyan]Lines:[/] {line_count:,}",
            border_style="green",
            title="Extract Output",
            box=box.HEAVY
        )
    )
    return jsonl, data_txt

def step_train_tokenizer(data_txt: Path, model_prefix: str, vocab_size: int, char_coverage: float, model_type: str):
    try:
        import sentencepiece as spm
    except ImportError:
        console.print("[red]sentencepiece belum terpasang. Jalankan: pip install sentencepiece[/]")
        sys.exit(1)

    start_ts = time.time()
    with Live(render_status(start_ts, 0, 0, str(data_txt), "Train Tokenizer"),
              refresh_per_second=5, console=console) as live:
        try:
            spm.SentencePieceTrainer.Train(
                input=str(data_txt),
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                character_coverage=char_coverage,
                model_type=model_type,
                input_sentence_size=2_000_000,
                shuffle_input_sentence=True
            )
            live.update(render_status(start_ts, 0, 0, str(data_txt), "Train Tokenizer"))
        except Exception:
            console.print("\n[red]Error saat Train Tokenizer![/]\n")
            console.print_exception()
            sys.exit(1)

    model_path = f"{model_prefix}.model"
    vocab_path = f"{model_prefix}.vocab"
    console.rule("[bold green]Tokenizer selesai")
    console.print(
        Panel.fit(
            f"[green]Model:[/] {model_path}\n"
            f"[green]Vocab:[/] {vocab_path}",
            border_style="green",
            title="Tokenizer Output",
            box=box.HEAVY
        )
    )
    return model_path, vocab_path

def main():
    ap = argparse.ArgumentParser(description="Wikipedia dump (.bz2) -> JSONL + data.txt -> SentencePiece tokenizer")
    ap.add_argument("--dump", default=DEFAULT_DUMP, help="Path ke file *pages-articles-multistream.xml.bz2")
    ap.add_argument("--out",  default=DEFAULT_OUT, help="Folder output untuk wiki.jsonl & data.txt")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX, help="Prefix output tokenizer (tanpa ekstensi)")
    ap.add_argument("--vocab", type=int, default=DEFAULT_VOCAB, help="Ukuran vocab (default 16000)")
    ap.add_argument("--coverage", type=float, default=DEFAULT_CHARCOVER, help="character_coverage (default 0.9995)")
    ap.add_argument("--model_type", default="bpe", choices=["bpe","unigram","char","word"], help="Tipe SentencePiece")
    ap.add_argument("--sample", type=int, default=None, help="Hanya proses N artikel pertama (tes cepat)")
    ap.add_argument("--skip_extract", action="store_true", help="Lewati extract (pakai data.txt yang sudah ada)")
    ap.add_argument("--skip_train", action="store_true", help="Lewati training tokenizer")

    args = ap.parse_args()

    dump = Path(args.dump)
    out_dir = Path(args.out)
    prefix = args.prefix

    if not args.skip_extract:
        if not dump.exists():
            console.print(f"[red]Dump tidak ditemukan:[/] {dump}")
            sys.exit(1)
        _, data_txt = step_extract(str(dump), out_dir, sample=args.sample, dedup=True)
    else:
        data_txt = out_dir / "data.txt"
        if not data_txt.exists():
            console.print(f"[red]data.txt tidak ditemukan:[/] {data_txt}  (Matikan --skip_extract atau pastikan file ada)")
            sys.exit(1)

    if not args.skip_train:
        step_train_tokenizer(data_txt, prefix, args.vocab, args.coverage, args.model_type)
    else:
        console.print("[yellow]Skip train tokenizer sesuai argumen.[/]")
        console.print(f"Gunakan file: {data_txt}")

if __name__ == "__main__":
    main()
