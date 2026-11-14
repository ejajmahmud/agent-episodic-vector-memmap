"""CLI for agent-episodic-vector-memmap."""
import argparse
import sys
from .core import VectorMemMapIndex

def main():
    parser = argparse.ArgumentParser(description="Agent Episodic Vector MemMap CLI")
    parser.add_argument("--demo", action="store_true", help="Run vector index search demo")
    args = parser.parse_args()

    if args.demo:
        idx = VectorMemMapIndex(dimension=4)
        idx.add("doc_git", [1.0, 0.0, 0.0, 0.0], {"topic": "version_control"})
        idx.add("doc_python", [0.0, 1.0, 0.0, 0.0], {"topic": "coding"})
        idx.add("doc_docker", [0.0, 0.0, 1.0, 0.0], {"topic": "containers"})

        query = [0.8, 0.6, 0.0, 0.0]
        results = idx.search(query, top_k=2)

        print(f"[+] Query Vector: {query}")
        print(f"[+] Top {len(results)} Nearest Memories:")
        for doc_id, score, meta in results:
            print(f"    - {doc_id} (sim={score:.4f}, meta={meta})")
    else:
        parser.print_help()
    sys.exit(0)

if __name__ == "__main__":
    main()
