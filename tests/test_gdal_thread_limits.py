import ast
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _thread_creation_options(relative_path, function_name):
    source_path = REPOSITORY_ROOT / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    return [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("NUM_THREADS=")
    ]


class GdalThreadLimitTest(unittest.TestCase):
    def test_instance_raster_compression_uses_one_gdal_thread(self):
        self.assertEqual(
            _thread_creation_options(
                "methods/main/PostprocHandler.py", "save_instance_raster"
            ),
            ["NUM_THREADS=1"],
        )

    def test_lclu_warp_creation_uses_one_gdal_thread(self):
        self.assertEqual(
            _thread_creation_options("methods/main/inference.py", "execute"),
            ["NUM_THREADS=1"],
        )


if __name__ == "__main__":
    unittest.main()
