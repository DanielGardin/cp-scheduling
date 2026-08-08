"""Script for checking inconsistencies between the `tests` structure and the source."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

SRC_DIR = PROJECT_ROOT / "cpscheduler"
TESTS_DIR = PROJECT_ROOT / "tests"

# Directories in tests/ allowed to exist without a matching src/ folder
ALLOWED_TEST_ONLY_DIRS = {"fixtures", "helpers", "mocks", "e2e", "test_utils"}

IGNORED_DIRS = {"__pycache__", ".git", ".pytest_cache", ".venv"}


def check_folder_alignment() -> None:
    """Check alignment between tests and source directory structure."""
    if not SRC_DIR.exists() or not TESTS_DIR.exists():
        print("Error: Both 'src/' and 'tests/' directories must exist.")
        sys.exit(1)

    orphaned_folders = []

    for test_folder in TESTS_DIR.rglob("**/*"):
        if any(part in IGNORED_DIRS for part in test_folder.parts):
            continue

        rel_path = test_folder.relative_to(TESTS_DIR)

        if rel_path.parts[0] in ALLOWED_TEST_ONLY_DIRS:
            continue

        expected_src_folder = SRC_DIR / rel_path

        if not expected_src_folder.exists() or not expected_src_folder.is_dir():
            orphaned_folders.append((test_folder, expected_src_folder))

    # --- Report ---
    print("=== Test Folder Alignment Check ===")

    if orphaned_folders:
        print(f"\nFound {len(orphaned_folders)} orphaned test folder(s):")
        for test_path, expected_src in orphaned_folders:
            print(f"  • {test_path}")
            print(f"    Missing source folder: {expected_src}\n")

        sys.exit(1)

    print("\nAll test folders align with current source directories!")
    sys.exit(0)


if __name__ == "__main__":
    check_folder_alignment()
