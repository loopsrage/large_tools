import asyncio
import os
import unittest
from asyncio import TaskGroup

from application_controller.app_controller import SimpleApp, run_app
from dotenv import load_dotenv
from fsspecc.base_fsspecfs.base_fsspecfs import FSBase
from qdrant_client import QdrantClient
from settings.helper import setting, restore

from qdrant_pdf.qdrant_pdf import PDFIndexQueue
from qdrantlib.qdrantlib import QdrantBGEM3

load_dotenv()
restore(os.getenv("ENV_FILE"))
pdfs = FSBase(filesystem="local", path="")
qdrant_client = QdrantBGEM3(QdrantClient(
    path="./qdrant_data"),
    model_cache_dir=setting("LocalAI", "fastembed_cache"))

class MyTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_something(self):
        app = PDFIndexQueue(qdrant_client, 1024, 1024, 10)
        async with TaskGroup() as tg:
            async def async_wrap():
                await run_app("PDFS", app.simple_app, "file/settings/settings.yaml")
            tg.create_task(async_wrap())
            app.init(pdfs, "./hts/tariffs")
            await app.simple_app.action_queues.wait_for_completions()
            print("\n[SUCCESS] 100% of pipeline data has been verified on disk. Exiting process...")
            return



async def test_somethinfag(self):
        query = f"""item:\n  name: Reamer Shoe for Casing Operations\n  material: UNKNOWN\n  mechanical_role: Casing Reamer Shoe for Wellbore Expansion and Stabilization\n  attributes:\n    casing_size: 139.7 mm (5.5 inches)\n    casing_size_fractional: 5-1/2\n    casing_weight_maximum: 26.0 lbs (38.69 kg)\n    casing_weight_minimum: 23.07 lbs (15.5 kg)\n    casing_weight_range: 15.5# - 26.0#\n    connection_up_type: BOX\n    cutting_structure: Tungsten Carbide\n    default_uom: each\n    generic_description: Reamer Shoe\n    long_description: >\n      Works order serial number controls individual bore size for each casing weight.\n      Circulation ports = 3, circulation port size = 30.0 mm (1.181 inches),\n      number of pads = 12, number of gauge blades = 6, gauge length = 173.0 mm (6.813 inches),\n      gauge protection = Tungsten Carbide brickettes, back pressure rating = 10,000 PSI,\n      overall length = 943.0 mm (37.126 inches), set down weight rating = 100,000 LBF\n    material_grade: P110\n    material_specification: Z refer to components\n    material_specification_supplements: Z refer to components\n    model_designation: DIAMONDBACK\n    nose_configuration: Eccentric Aluminum\n    optional_valve: Integral SS3\n    outside_diameter_maximum: 203.2 mm (8.0 inches)\n    pdc_drillability: PDC drillable\n    primary_legacy: RD054080CV\n    reference_documents: D001176110\n    total_flow_area: 3.14 (square inches) / 2025.80 (square mm)"""
        resp = qdrant_client.fuzzy_query(
            "pdfs",
            query,
            fuzzy_threshold=10.,
            dense_threshold=.01,
            sparse_threshold=.01)
        for text, score in resp:
            if score > 20.0:
                print(f"[{score:.1f}% Match] {text}")
        has_correct_chapter = any("8431" in text or "7304" in text for text, score in resp if score > 20)
        self.assertTrue(has_correct_chapter, "Failed to locate relevant HTS chapters in vector storage context.")


if __name__ == '__main__':
    unittest.main()
