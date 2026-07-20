"""Build the medquad_chroma vector index from the MedQuAD dataset.

Clones MedQuAD (if not already present), parses every QAPair out of its XML
files, embeds each Q&A pair with OpenAI's text-embedding-3-small, and
persists the result to medquad_chroma/. Required once before running the
agent or building the Docker image.

Usage:
    uv run python scripts/build_vector_db.py [--rebuild]
"""

from __future__ import annotations

import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import git
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
MEDQUAD_DIR = BASE_DIR / "MedQuAD"
MEDQUAD_REPO_URL = "https://github.com/abachaa/MedQuAD.git"
CHROMA_DIR = BASE_DIR / "medquad_chroma"
EMBEDDING_MODEL = "text-embedding-3-small"


def ensure_dataset() -> None:
    if MEDQUAD_DIR.exists():
        print(f"MedQuAD dataset already present at {MEDQUAD_DIR}")
        return
    print(f"Cloning MedQuAD into {MEDQUAD_DIR}...")
    git.Repo.clone_from(MEDQUAD_REPO_URL, str(MEDQUAD_DIR))


def load_documents() -> list[Document]:
    documents: list[Document] = []
    folders = [f for f in sorted(MEDQUAD_DIR.iterdir()) if f.is_dir()]
    for folder in folders:
        xml_files = sorted(folder.glob("*.xml"))
        for xml_file in tqdm(xml_files, desc=folder.name):
            try:
                tree = ET.parse(xml_file)
            except ET.ParseError as exc:
                print(f"Skipping unparsable file {xml_file}: {exc}")
                continue
            for qa in tree.getroot().findall(".//QAPair"):
                question = qa.findtext("Question")
                answer = qa.findtext("Answer")
                if question and answer:
                    documents.append(
                        Document(
                            page_content=f"Question: {question}\nAnswer: {answer}",
                            metadata={"source": folder.name},
                        )
                    )
    return documents


def build_index(rebuild: bool) -> None:
    if CHROMA_DIR.exists():
        if not rebuild:
            print(f"{CHROMA_DIR} already exists. Pass --rebuild to regenerate it.")
            return
        print(f"Removing existing index at {CHROMA_DIR}...")
        shutil.rmtree(CHROMA_DIR)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise SystemExit("OPENAI_API_KEY is not set (check your .env file)")

    ensure_dataset()

    print("Parsing MedQuAD XML files...")
    documents = load_documents()
    print(f"Loaded {len(documents)} question-answer pairs")

    print(
        f"Embedding {len(documents)} documents and persisting to {CHROMA_DIR} "
        "(this calls the OpenAI embeddings API and may take a while)..."
    )
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=openai_api_key)
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print("Done.")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and regenerate the index if it already exists.",
    )
    args = parser.parse_args()
    build_index(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
