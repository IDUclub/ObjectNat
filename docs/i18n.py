"""Translations of the documentation: keep the catalogues in step with the English sources.

The English text (pages, notebook text cells, docstrings shown by autodoc) is the source; every translation lives
in ``docs/locale/<language>/LC_MESSAGES/<page>.po``, one catalogue per page.

    python docs/i18n.py update   # extract the text, add new and changed messages to the catalogues
    python docs/i18n.py check    # exit with 1 while a message is missing, untranslated or fuzzy

After ``update``, translate the entries with an empty ``msgstr`` and review the ones marked ``#, fuzzy`` (text
that changed in English: fix the translation, then delete the flag). Fuzzy and empty entries fall back to
English in the built site, and ``check`` (run in CI) lists them.

A parameter description with a nested list (``Text:  - item - item`` in the catalogue) also appears as its
separate parts. ``update`` copies such a container message unchanged, so that Sphinx keeps its structure and
translates the parts; translate the parts.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from babel.messages.catalog import Catalog
from babel.messages.pofile import read_po, write_po

DOCS = Path(__file__).resolve().parent
LOCALE = DOCS / "locale"
LANGUAGES = ("ru",)


def extract(target: Path) -> None:
    """Message templates (.pot) of every page, from Sphinx's gettext builder."""
    command = [sys.executable, "-m", "sphinx", "-q", "-E", "-b", "gettext", str(DOCS), str(target)]
    subprocess.run(command, check=True)


def pairs(templates: Path, language: str):
    for template in sorted(templates.rglob("*.pot")):
        yield template, LOCALE / language / "LC_MESSAGES" / template.relative_to(templates).with_suffix(".po")


def read(path: Path, locale=None) -> Catalog:
    with path.open("rb") as file:
        return read_po(file, locale=locale)


def is_container(message_id) -> bool:
    """A parameter description with a nested list, extracted as a whole besides its parts."""
    return "  - " in str(message_id)


def update(templates: Path) -> None:
    for language in LANGUAGES:
        kept = set()
        for template_path, po_path in pairs(templates, language):
            template = read(template_path)
            if po_path.exists():
                catalog = read(po_path, language)
            else:
                catalog = Catalog(
                    locale=language,
                    project="ObjectNat",
                    copyright_holder="IDU, ITMO University",
                    msgid_bugs_address="https://github.com/IDUclub/ObjectNat/issues",
                    header_comment=f"# {language} translation of the ObjectNat documentation, see docs/i18n.py",
                    fuzzy=False,
                )
            catalog.update(template, update_creation_date=False)
            for message in catalog:
                if message.id and is_container(message.id):
                    # an unchanged message is left alone by Sphinx, which then translates the parts
                    message.string = message.id
                    message.flags.discard("fuzzy")
            po_path.parent.mkdir(parents=True, exist_ok=True)
            with po_path.open("wb") as file:
                write_po(file, catalog, width=0, ignore_obsolete=True, include_previous=True, sort_output=False)
            kept.add(po_path)
            todo = sum(1 for m in catalog if m.id and (not m.string or m.fuzzy))
            if todo:
                print(f"{po_path.relative_to(DOCS)}: {todo} to translate")
        for stale in (LOCALE / language).rglob("*.po"):
            if stale not in kept:
                stale.unlink()
                print(f"removed {stale.relative_to(DOCS)} (the page no longer exists)")


def check(templates: Path) -> int:
    problems = 0
    for language in LANGUAGES:
        for template_path, po_path in pairs(templates, language):
            catalog = read(po_path, language) if po_path.exists() else None
            for message in read(template_path):
                if not message.id:
                    continue
                entry = catalog.get(message.id, message.context) if catalog is not None else None
                if entry is None:
                    status = "missing"
                elif entry.fuzzy:
                    status = "fuzzy"
                elif not entry.string:
                    status = "untranslated"
                elif is_container(message.id) and entry.string != message.id:
                    status = "container must stay unchanged (translate its parts)"
                else:
                    continue
                problems += 1
                print(f"{po_path.relative_to(DOCS)}: {status}: {str(message.id)[:100]}")
    if problems:
        print(f"\n{problems} message(s) need a translation: run `python docs/i18n.py update`, translate them, rerun.")
    else:
        print("All messages are translated.")
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["update", "check"])
    args = parser.parse_args()
    templates = Path(tempfile.mkdtemp(prefix="objectnat-gettext-"))
    try:
        extract(templates)
        if args.command == "update":
            update(templates)
            return 0
        return check(templates)
    finally:
        shutil.rmtree(templates, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
