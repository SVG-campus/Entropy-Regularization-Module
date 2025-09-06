import os, pytest

MAIN_PDF = "Entropy_Regularization_Module.pdf"
TEST_PDF = "Entropy_Regularization_Module(Test).pdf"
TEST_NB  = "Entropy_Regularization_Module_in_Hilbert_Spaces(Test).ipynb"

def test_pdfs_and_notebook_exist_or_skip():
    missing = [p for p in [MAIN_PDF, TEST_PDF, TEST_NB] if not os.path.exists(p)]
    if missing:
        pytest.skip("Missing research artifacts: " + ", ".join(missing))
    for p in [MAIN_PDF, TEST_PDF]:
        if os.path.exists(p):
            assert os.path.getsize(p) > 0, f"{p} exists but is empty"
