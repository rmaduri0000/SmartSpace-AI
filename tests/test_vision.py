"""
Unit Tests for SmartSpace AI Vision Module & State Extractor
"""
import unittest
from PIL import Image

from vision.detector import VisionStateExtractor
from config import TARGET_CLASSES

class TestVisionModule(unittest.TestCase):
    def setUp(self):
        self.extractor = VisionStateExtractor()

    def test_vision_pipeline(self):
        # Create a blank test image
        img = Image.new("RGB", (640, 640), (230, 230, 230))
        result = self.extractor.process_room_image(img)
        
        self.assertTrue(result["success"])
        self.assertGreater(result["detected_count"], 0)
        self.assertIn("annotated_image", result)
        self.assertIn("room_state", result)
        
        # Verify room state contains required fields
        room_state = result["room_state"]
        self.assertIn("dimensions", room_state)
        self.assertIn("door", room_state)
        self.assertIn("windows", room_state)
        self.assertIn("initial_furniture", room_state)

if __name__ == '__main__':
    unittest.main()
