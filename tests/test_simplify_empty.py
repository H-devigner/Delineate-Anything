import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from osgeo import gdal, ogr, osr


gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")


def _install_optional_dependency_shims():
    """Let the GDAL-only tests import worker modules without their runtimes."""
    try:
        import cv2  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["cv2"] = types.ModuleType("cv2")

    try:
        import numba  # noqa: F401
    except ModuleNotFoundError:
        numba = types.ModuleType("numba")

        def njit(*decorator_args, **decorator_kwargs):
            if (
                len(decorator_args) == 1
                and callable(decorator_args[0])
                and not decorator_kwargs
            ):
                return decorator_args[0]
            return lambda function: function

        numba.njit = njit
        numba.prange = range
        sys.modules["numba"] = numba

    try:
        import tqdm  # noqa: F401
    except ModuleNotFoundError:
        tqdm = types.ModuleType("tqdm")
        tqdm.tqdm = lambda iterable=None, *args, **kwargs: iterable
        sys.modules["tqdm"] = tqdm


_install_optional_dependency_shims()

from simplification import simplify as simplify_module


class EmptySimplificationTest(unittest.TestCase):
    layer_name = "fields"
    epsilon = 1.25

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.source_path = root / "source.gpkg"
        self.destination_path = root / "destination.gpkg"

    def _create_source(self, *, add_feature=False):
        driver = ogr.GetDriverByName("GPKG")
        dataset = driver.CreateDataSource(str(self.source_path))
        self.assertIsNotNone(dataset)

        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromEPSG(32635)
        layer = dataset.CreateLayer(
            self.layer_name,
            spatial_reference,
            geom_type=ogr.wkbMultiPolygon,
        )
        self.assertIsNotNone(layer)

        identifier = ogr.FieldDefn("field_id", ogr.OFTInteger64)
        label = ogr.FieldDefn("label", ogr.OFTString)
        label.SetWidth(48)
        self.assertEqual(layer.CreateField(identifier), ogr.OGRERR_NONE)
        self.assertEqual(layer.CreateField(label), ogr.OGRERR_NONE)
        dataset.SetMetadata({"SOURCE_SCENE": "skysat-test-scene"})

        if add_feature:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for x, y in ((0, 0), (10, 0), (10, 10), (0, 10), (0, 0)):
                ring.AddPoint_2D(x, y)
            polygon = ogr.Geometry(ogr.wkbPolygon)
            polygon.AddGeometry(ring)
            geometry = ogr.Geometry(ogr.wkbMultiPolygon)
            geometry.AddGeometry(polygon)
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("field_id", 7)
            feature.SetField("label", "test")
            feature.SetGeometry(geometry)
            self.assertEqual(layer.CreateFeature(feature), ogr.OGRERR_NONE)

        dataset.FlushCache()
        layer = None
        dataset = None

    def _run_simplification(self):
        simplify_module.simplify_internal(
            str(self.source_path),
            self.layer_name,
            str(self.destination_path),
            self.layer_name,
            densify_step=[1.0, 1.0],
            epsilon=self.epsilon,
            region_size=[32, 32],
            num_workers=2,
            fid_column="field_id",
        )

    @contextmanager
    def _forbid_multiprocessing(self):
        failure = AssertionError("multiprocessing workers must not start")
        with mock.patch.multiple(
            simplify_module.multiprocessing,
            Queue=mock.Mock(side_effect=failure),
            Value=mock.Mock(side_effect=failure),
            Event=mock.Mock(side_effect=failure),
        ), mock.patch.object(
            simplify_module.ReadWorker, "start", side_effect=failure
        ), mock.patch.object(
            simplify_module.SimplificationWorker, "start", side_effect=failure
        ), mock.patch.object(
            simplify_module.WriteWorker, "start", side_effect=failure
        ):
            yield

    @staticmethod
    def _field_signature(layer):
        definition = layer.GetLayerDefn()
        return [
            (
                definition.GetFieldDefn(index).GetName(),
                definition.GetFieldDefn(index).GetType(),
                definition.GetFieldDefn(index).GetWidth(),
                definition.GetFieldDefn(index).GetPrecision(),
            )
            for index in range(definition.GetFieldCount())
        ]

    def test_empty_source_creates_schema_complete_destination_without_workers(self):
        self._create_source()

        with self._forbid_multiprocessing():
            self._run_simplification()

        source = ogr.Open(str(self.source_path), 0)
        destination = ogr.Open(str(self.destination_path), 0)
        self.assertIsNotNone(destination)
        source_layer = source.GetLayerByName(self.layer_name)
        destination_layer = destination.GetLayerByName(self.layer_name)
        self.assertIsNotNone(destination_layer)
        self.assertEqual(destination_layer.GetFeatureCount(), 0)
        self.assertEqual(destination_layer.GetGeomType(), source_layer.GetGeomType())
        self.assertTrue(
            bool(destination_layer.GetSpatialRef().IsSame(source_layer.GetSpatialRef()))
        )
        self.assertEqual(
            self._field_signature(destination_layer),
            self._field_signature(source_layer),
        )
        metadata = destination.GetMetadata()
        self.assertEqual(metadata["SOURCE_SCENE"], "skysat-test-scene")
        self.assertAlmostEqual(
            float(metadata["SIMPLIFICATION_EPSILON"]), self.epsilon
        )

    def test_empty_source_returns_before_requesting_extent(self):
        self._create_source()

        with mock.patch.object(
            ogr.Layer,
            "GetExtent",
            side_effect=AssertionError("empty layers have no processing extent"),
        ), self._forbid_multiprocessing():
            self._run_simplification()

    def test_null_extent_returns_without_starting_workers(self):
        self._create_source(add_feature=True)

        with mock.patch.object(
            ogr.Layer, "GetExtent", return_value=None
        ), self._forbid_multiprocessing():
            self._run_simplification()

        destination = ogr.Open(str(self.destination_path), 0)
        self.assertIsNotNone(destination)
        layer = destination.GetLayerByName(self.layer_name)
        self.assertIsNotNone(layer)
        self.assertEqual(layer.GetFeatureCount(), 0)

    def test_nonempty_source_still_requests_an_extent(self):
        self._create_source(add_feature=True)

        class ExtentRequested(Exception):
            pass

        with mock.patch.object(
            ogr.Layer,
            "GetExtent",
            side_effect=ExtentRequested,
        ), self.assertRaises(ExtentRequested):
            self._run_simplification()


if __name__ == "__main__":
    unittest.main()
